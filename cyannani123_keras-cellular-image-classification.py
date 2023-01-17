import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys, random,copy

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate

from keras.utils import np_utils, Sequence

from PIL import Image

from PIL import ImageFilter

from sklearn.model_selection import train_test_split

!pip install efficientnet

import efficientnet

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
test_data = pd.read_csv("../input/recursion-cellular-image-classification/test.csv")

print("Shape of test_data:", test_data.shape)

test_data.head()
train_data = pd.read_csv("../input/recursion-cellular-image-classification/train.csv")

print("Shape of train_data:", train_data.shape)

train_data.head()
def get_input(id_code,site,train=True):

    if train==True:

        base_path = '../input/recursion-cellular-image-classification-128/train/train'

    else:

        base_path = '../input/recursion-cellular-image-classification-128/test/test'

    img = Image.open(base_path+"/"+id_code+"_s"+str(site)+".jpeg")

    return(img)
class ImgGen (Sequence):

    def __init__(self, label_file, batch_size = 32,preprocess=(lambda x: x),train=True,shuffle=False):

        if shuffle==True:

            self.label_file = label_file.sample(frac=1)

        else:

            self.label_file = label_file

        self.batch_size = batch_size

        self.preprocess = preprocess

        self.train = train

        self.x = list(self.label_file.index)

        if self.train==True:

            self.y = list(self.label_file[self.label_file.columns[0]])



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]



        if self.train==True:

            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            

            x1 = [np.array(self.preprocess(get_input(id_code,1))).reshape(128,128,3) / 255 for id_code in batch_x] 

            x2 = [np.array(self.preprocess(get_input(id_code,2))).reshape(128,128,3) / 255 for id_code in batch_x]

            y = [sirna for sirna in batch_y]

            return [np.array(x1),np.array(x2)], np.array(y)

        else:

            x1 = [np.array(self.preprocess(get_input(id_code,1,train=False))).reshape(128,128,3) / 255 for id_code in batch_x] 

            x2 = [np.array(self.preprocess(get_input(id_code,2,train=False))).reshape(128,128,3) / 255 for id_code in batch_x] 

            return [np.array(x1),np.array(x2)]
def augment(image):

    random_transform = random.randint(-1,4)

    if random_transform==0:

        image = image.rotate(random.randint(-5,5))

    if random_transform==1:

        image = image.filter(ImageFilter.GaussianBlur(radius=1))

    if random_transform==2:

        image = image.filter(ImageFilter.RankFilter(size=3, rank=1))

    if random_transform==3:

        image = image.filter(ImageFilter.MedianFilter(size=3))

    if random_transform==4:

        image = image.filter(ImageFilter.MaxFilter(size=3))

    return image
def create_model():

    effnet = efficientnet.EfficientNetB1(weights='imagenet',include_top=False,input_shape=(128, 128, 3))

    site1 = Input(shape=(128,128,3))

    site2 = Input(shape=(128,128,3))

    x = effnet(site1)

    x = GlobalAveragePooling2D()(x)

    x = Model(inputs=site1, outputs=x)

    y = effnet(site2)

    y = GlobalAveragePooling2D()(y)

    y = Model(inputs=site2, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dropout(0.5)(combined)

    z = Dense(1108, activation='softmax')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

    model.summary()

    

    return model



model = create_model()
phases = ['All','HEPG2','HUVEC','RPE','U2OS']

batch_size = 128

test_size = 0.025



for phase in phases:

    

    print('Start phase %s.' % (phase))

    

    filepath = 'ModelCheckpoint_'+phase+'.h5'

    print("Set filepath to %s." % (filepath))

    

    callback = [

        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        ]

    

    if phase != 'All':

        model.load_weights('ModelCheckpoint_All.h5')

        print("Successfully loaded weights of ModelCheckpoint_All.h5")

    

    if phase != 'All':

        label_file_source = train_data[train_data['id_code'].str.contains(phase)]

        print("Successfully created label_file_source for phase %s." % (phase))

    else:

        label_file_source = train_data

        print("Successfully created label_file_source for phase All.")

    

    label_file = pd.DataFrame(index=label_file_source['id_code'],data=list(label_file_source['sirna']),columns=['sirna'])

    train, val = train_test_split(label_file, test_size=test_size)

    train_gen = ImgGen(train,batch_size=batch_size,shuffle=True,preprocess=augment)

    val_gen = ImgGen(val,batch_size=batch_size,shuffle=True,preprocess=augment)

    

    history = model.fit_generator(train_gen, 

                              steps_per_epoch=len(train)//batch_size, 

                              epochs=25, 

                              verbose=1, 

                              validation_data=val_gen,

                              validation_steps=len(val)//batch_size,

                              callbacks=callback

                             )
submission = pd.DataFrame()

id_codes = []



for phase in phases[1:]:

    test_label_file = pd.DataFrame(index=(test_data[test_data['id_code'].str.contains(phase)]['id_code']))

    

    model.load_weights('ModelCheckpoint_'+phase+'.h5')

    print("Successfully loaded weights of ModelCheckpoint_%s.h5" % (phase))

    test_generator = ImgGen(test_label_file,batch_size=batch_size,train=False)

    if phase == phases[1]:

        predictions = model.predict_generator(test_generator,verbose=1)

    else:

        predictions = np.append(predictions,model.predict_generator(test_generator,verbose=1), axis=0)

    id_codes += list(test_label_file.index)

    

submission['id_code'] = id_codes

submission['sirna'] = predictions.argmax(axis=-1)

submission.to_csv("submission.csv",index=False)



print(pd.read_csv("submission.csv"))