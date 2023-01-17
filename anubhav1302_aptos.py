import numpy as np

import pandas as pd

import os

import albumentations as ab

import matplotlib.pyplot as plt

import cv2

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import seaborn as sns

from sklearn.utils import class_weight,shuffle

import keras as K

from keras.utils import to_categorical

from keras.models import Model

from keras.regularizers import l1

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback,ModelCheckpoint,ReduceLROnPlateau

import imgaug.augmenters as iaa

import imgaug as ia
train_df_path='../input/aptos2019-blindness-detection/train.csv'

train_img_path='../input/aptos2019-blindness-detection/train_images/'

test_df_path='../input/aptos2019-blindness-detection/test.csv'

test_images_path='../input/aptos2019-blindness-detection/test_images/'
#Hyperparameters

SIZE=224

batch_size=35
train_df=pd.read_csv(train_df_path)

test_df=pd.read_csv(test_df_path)
print('Train Data shape: {}'.format(train_df.shape))

print('Test Data shape: {}'.format(test_df.shape))
class_dict={0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'}
new_id_col=[i+'.png' for i in train_df['id_code']]

new_diagnosis=[class_dict[i] for i in train_df['diagnosis']]

train_df['new_id_col']=new_id_col

train_df['new_diagnosis']=new_diagnosis
train_df.head()
sns.distplot(train_df['diagnosis'])
#Print some samples

#without Augmentation

w=h=5

fig=plt.figure(figsize=(10,10))

rows,cols=3,3

for i in range(1,rows*cols+1):

    img=cv2.imread(os.path.join(train_img_path,train_df.loc[i-1,'new_id_col']))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    fig.add_subplot(rows,cols,i)

    plt.title('Diagnosis: {}'.format(train_df.loc[i-1,'new_diagnosis']))

    plt.imshow(img)

plt.show()
class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train_df['diagnosis']),

                                                   train_df['diagnosis'])
#Split dataset

train_data,val_data=train_test_split(train_df,test_size=0.15)

train_data,val_data=train_data.reset_index(drop=True),val_data.reset_index(drop=True)
print('Train Data Shape: {}'.format(train_data.shape))

print('Val Data Shape: {}'.format(val_data.shape))
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(

    [# apply the following augmenters to most images

        iaa.Fliplr(0.5), # horizontally flip 50% of all images

        iaa.Flipud(0.4), # vertically flip 20% of all images

        sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5))),

        iaa.Affine(

            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

            rotate=(-45, 45), # rotate by -45 to +45 degrees

            shear=(-16, 16), # shear by -16 to +16 degrees

           order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

            cval=(0, 255), # if mode is constant, use a cval between 0 and 255

            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

        )

    ])                  
class My_Generator(K.utils.Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size,path, is_train=True,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        self.path=path

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread(os.path.join(self.path,sample))

            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (SIZE, SIZE))

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        batch_y=to_categorical(batch_y,num_classes=5)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread(os.path.join(self.path,sample))

            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        batch_y=to_categorical(batch_y,num_classes=5)

        return batch_images, batch_y
from keras.callbacks import Callback

class QWKEvaluation(Callback):

    def __init__(self, validation_data=(), batch_size=batch_size, interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.batch_size = batch_size

        self.valid_generator, self.y_val = validation_data

        self.history = []



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict_generator(generator=self.valid_generator,

                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),

                                                  workers=1, use_multiprocessing=False,

                                                  verbose=1)

            def flatten(y):

                return np.argmax(y, axis=1).reshape(-1)

            

            score = cohen_kappa_score(self.y_val,

                                      flatten(y_pred),

                                      labels=[0,1,2,3,4],

                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))

            self.history.append(score)

            if score >= max(self.history):

                print('saving checkpoint: ', score)

                self.model.save('../working/classifier_6.h5')



train_generator=My_Generator(np.asarray(train_data['new_id_col']),np.asarray(train_data['diagnosis']),

                            batch_size=batch_size,path=train_img_path,is_train=True,mix=False,augment=True)



val_generator=My_Generator(np.asarray(val_data['new_id_col']),np.asarray(val_data['diagnosis']),

                            batch_size=batch_size,path=train_img_path,is_train=False,mix=False,augment=False)
qwk = QWKEvaluation(validation_data=(val_generator, np.asarray(val_data['diagnosis'])),

                    batch_size=batch_size, interval=1)
reg = l1(0.001)
base_layer=K.applications.densenet.DenseNet169(input_shape=(224,224,3),include_top=False,

                                           weights='../input/densenet-keras/DenseNet-BC-169-32-no-top.h5')

for layer in base_layer.layers:

    layer.trainable=True



lstm_=K.layers.Reshape([7*7,1664])(base_layer.output)

lstm_=K.layers.Bidirectional(K.layers.LSTM(832,recurrent_dropout=0.2))(lstm_)

x=K.layers.Dense(1024,activation='relu',activity_regularizer=reg)(lstm_)

x=K.layers.Dropout(0.3)(x)

out=K.layers.Dense(5,activation='softmax')(x)
model=Model(base_layer.input,out)

model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001))
rp=ReduceLROnPlateau(monitor='val_loss',factor=0.2,min_lr=0.00000001,patience=3,verbose=1)

mc=ModelCheckpoint('../working/classifier_6.h5',monitor='val_loss',mode='min',period=1,save_best_only=True)
history=model.fit_generator(train_generator,epochs=30,steps_per_epoch=int(np.ceil(len(train_data)/batch_size)),

                    validation_data=val_generator,

                            validation_steps=int(np.ceil(len(val_data)/batch_size)),callbacks=[qwk,rp]

                            ,class_weight=class_weights)
loss = history.history['loss']

val_loss = history.history['val_loss']

score=qwk.history

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'b',color='red',label='Training Loss')

plt.plot(epochs,val_loss,'b',color='blue',label='Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()

plt.figure()

plt.plot(epochs,score,'b',color='red',label='Validation Kappa')

plt.legend()

plt.figure()

plt.show()
#delete unnecessary data

del train_generator,val_generator,train_data,val_data
#Load Model

model.load_weights('../working/classifier_6.h5')

print('Weights Restored')
#Test Predictions

test_df=pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

predictions=[]

for id_ in tqdm(test_df['id_code']):

    img = cv2.imread(os.path.join(test_images_path,id_+'.png'))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (SIZE, SIZE))

    img=img/255.0

    img=np.expand_dims(img,0)

    predictions.extend(model.predict(img))
predictions=np.argmax(predictions,1)

sub=pd.DataFrame({'id_code':test_df['id_code'],'diagnosis':predictions})

sub.to_csv('submission.csv',index=False)