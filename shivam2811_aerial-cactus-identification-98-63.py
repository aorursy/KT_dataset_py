import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import matplotlib.pyplot as plt

!pip install opencv-contrib-python

import cv2

import os

print(os.listdir("../input"))

from tqdm import tqdm, tqdm_notebook

train_images= os.listdir("../input/train/train/")

train_dir = "../input/train/train/"

test_dir = "../input/test/test/"

test_images= os.listdir("../input/test/test/")

label_df = pd.read_csv('../input/train.csv')

train_labels=label_df['has_cactus']
X_tr = []

Y_tr = []

imges = label_df['id'].values

for img_id in tqdm_notebook(imges):

    X_tr.append(cv2.imread(train_dir + img_id))    

    Y_tr.append(label_df[label_df['id'] == img_id]['has_cactus'].values[0])  

X_tr = np.asarray(X_tr)

X_tr = X_tr.astype('float32')

X_tr /= 255

#Y_tr = np.asarray(Y_tr)
from keras.utils import to_categorical

Y_tr = to_categorical(train_labels, num_classes =2)
X_tr.shape
from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense,Dropout, Flatten, GlobalAveragePooling2D



resnet_weights_path ='imagenet'

def model():

    my_new_model = Sequential()

    my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

    my_new_model.add(Dense(256, activation='relu'))

    #my_new_model.add(Dropout(0.5))

    my_new_model.add(Dense(2, activation='softmax'))

    # Say not to train first layer (ResNet) model. It is already trained

    my_new_model.layers[0].trainable = False



        
from PIL import Image

from tqdm import tqdm



from keras.preprocessing.image import ImageDataGenerator



def load_data(dataframe=None, batch_size=16, mode='categorical'):

    if dataframe is None:

        dataframe = pd.read_csv('../input/train.csv')

    dataframe['has_cactus'] = dataframe['has_cactus'].apply(str)

    gen = ImageDataGenerator(rescale=1./255., validation_split=0.1, horizontal_flip=True, vertical_flip=True)



    trainGen = gen.flow_from_dataframe(dataframe, directory='../input/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

        class_mode=mode, batch_size=batch_size, shuffle=True, subset='training')

    testGen = gen.flow_from_dataframe(dataframe, directory='../input/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

        class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')

    return trainGen, testGen
#my_new_model.fit(X_tr,Y_tr,validation_split=0.1,epochs=500,batch_size=32,verbose = 2)

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam, SGD

def m():

    

    trainGen, valGen = load_data(batch_size=32)

    

    my_new_model = model()

    my_new_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5, verbose=1)]

    my_new_model.fit_generator(trainGen, steps_per_epoch=4922, epochs=3, validation_data=valGen,validation_steps=493, shuffle=True,callbacks=cbs)

    return my_new_model

model =m()
for layer in my_new_model.layers:

    weights = layer.get_weights() 
my_new_model.save_weights("my_weights.h5")
sub_df = pd.read_csv('../input/sample_submission.csv')
X_ts = []

imges = sub_df['id'].values

for img_id in tqdm_notebook(imges):

    X_ts.append(cv2.imread(test_dir + img_id))    

X_ts = np.asarray(X_ts)

X_ts = X_ts.astype('float32')

X_ts /= 255

X_ts.shape

print(results)
results= np.empty((sub_df.shape[0],))

for n in tqdm(range(0,sub_df.shape[0])):

    results[n] = my_new_model.predict(X_ts[n].reshape((1, 32, 32, 3)))[0][1]

    #print (results)

#results = np.argmax(results,axis = 1)
f=sub_df['id']

results = pd.DataFrame(results,columns=['has_cactus'])



#results['has_cactus'] = results['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

results.head()
results.head()
submission = pd.concat([pd.Series(f,name ='id'),results],axis = 1)



submission.to_csv("samplesubmission.csv",index=False)
submission.head()