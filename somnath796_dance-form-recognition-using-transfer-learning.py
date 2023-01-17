# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

import numpy as np

import seaborn as sns

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt
train=pd.read_csv('/kaggle/input/indian-dance-form-recognition/dataset/train.csv')

test=pd.read_csv('/kaggle/input/indian-dance-form-recognition/dataset/test.csv')

train.head()
test.head()
print(train['target'].unique())
Class_map={'manipuri':0, 'bharatanatyam':1, 'odissi':2 ,'kathakali':3, 'kathak':4, 'sattriya':5,

 'kuchipudi':6, 'mohiniyattam':7}

inverse_map={0:'manipuri', 1:'bharatanatyam', 2:'odissi' ,3:'kathakali',4: 'kathak', 5:'sattriya',

 6:'kuchipudi', 7:'mohiniyattam'}

train['target']=train['target'].map(Class_map)
train.head()
img_h,img_w= (224,224)
train_img=[]

train_label=[]

j=0

path='/kaggle/input/indian-dance-form-recognition/dataset/train'

for i in tqdm(train['Image']):

    final_path=os.path.join(path,i)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(img_h,img_w))

    img=img.astype('float32')

    train_img.append(img)

    train_label.append(train['target'][j])

    j=j+1
from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(train_img, train_label, test_size=0.30, shuffle= True)
test_img=[]

path='/kaggle/input/indian-dance-form-recognition/dataset/test'

for i in tqdm(test['Image']):

    final_path=os.path.join(path,i)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(img_h,img_w))

    img=img.astype('float32')

    test_img.append(img)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,# divide each input by its std

        rescale=1./255,

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.3, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images



test_datagen= ImageDataGenerator(rescale=1./255)

valid_datagen= ImageDataGenerator(rescale=1./255)

train_datagen.fit(x_train)

test_datagen.fit(test_img)

valid_datagen.fit(x_valid)
train_img=np.array(train_img)

x_train= np.array(x_train)

x_valid= np.array(x_valid)

y_train= np.array(y_train)

y_valid= np.array(y_valid)

test_img=np.array(test_img)

train_label=np.array(train_label)

print("Shape of training data=",x_train.shape," and shape of labels of training data= ",y_train.shape)

print("Shape of validation data=",x_valid.shape," and shape of labels of validation data= ",y_valid.shape)

print("Shape of test data=",test_img.shape)
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.layers import Dropout



base_model= InceptionResNetV2(include_top=False, weights='imagenet', 

                              input_tensor=None, input_shape=(img_h,img_w,3), pooling='avg')
'''for layer in base_model.layers[:-10]:

    layer.trainable=False'''

base_model.trainable=False

    

base_model.summary()
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.callbacks import ReduceLROnPlateau





model=Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dropout(0.4))

model.add(BatchNormalization())



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



#model.add(Dense(256, activation='relu'))

#model.add(Dropout(0.2))

#model.add(BatchNormalization())



model.add(Dense(8,activation='softmax'))





from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop





#NASnet.trainable=False



reduce_learning_rate = ReduceLROnPlateau(monitor='loss',

                                         factor=0.1,

                                         patience=2,

                                         cooldown=2,

                                         min_lr=0.00001,

                                         verbose=1)



callbacks = [reduce_learning_rate]

    





model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()
model.fit(train_datagen.flow(x_train, to_categorical(y_train,8), batch_size=32),

                    epochs=10,

          callbacks=callbacks,

          validation_data= valid_datagen.flow(x_valid, to_categorical(y_valid,8), batch_size=32),

          verbose=1

             )
from tensorflow.keras.applications.resnet import ResNet50

base_model_2= ResNet50(include_top=False, weights='imagenet',input_shape=(img_h,img_w,3), pooling='max')



'''for layer in base_model_2.layers[:-3]:

    layer.trainable=False'''

base_model_2.trainable=False

    

model_2=Sequential()

model_2.add(base_model_2)

model_2.add(Flatten())

model_2.add(Dropout(0.4))

model_2.add(BatchNormalization())



model_2.add(Dense(768, activation='relu'))

model_2.add(Dropout(0.2))

model_2.add(BatchNormalization())



model_2.add(Dense(256, activation='relu'))

model_2.add(Dropout(0.1))

model_2.add(BatchNormalization())





model_2.add(Dense(8,activation='softmax'))



model_2.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_2.summary()
model_2.fit(train_datagen.flow(x_train, to_categorical(y_train,8), batch_size=32),

                    epochs=10,

          callbacks=callbacks,

          validation_data= valid_datagen.flow(x_valid, to_categorical(y_valid,8), batch_size=32),

          verbose=1

             )
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input

base_model_3=VGG19(include_top=False, weights='imagenet',input_shape=(img_h,img_w,3), pooling='max')



for layer in base_model_3.layers[:-4]:

    layer.trainable=False

#base_model_3.trainable=False

    

model_3=Sequential()

model_3.add(base_model_3)

model_3.add(Flatten())



model_3.add(BatchNormalization())

model_3.add(Dropout(0.2))





model_3.add(Dense(256, activation='relu'))

model_3.add(BatchNormalization())





model_3.add(Dense(64, activation='relu'))

model_3.add(BatchNormalization())

model_3.add(Dropout(0.2))







model_3.add(Dense(8,activation='softmax'))



model_3.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_3.summary()
model_3.fit(train_datagen.flow(x_train, to_categorical(y_train,8), batch_size=32),

                    epochs=20,

          callbacks=callbacks,

          validation_data= valid_datagen.flow(x_valid, to_categorical(y_valid,8), batch_size=32),

          verbose=1

             )
from tensorflow.keras import Input

from tensorflow.keras.layers import concatenate

def stacking_ensemble(members,input_shape,n_classes):

  commonInput = Input(shape=input_shape)

  out=[]



  for model in members:

    #model._name= model._name+"test"+ str(members.index(model)+1)

    model._name= model.get_layer(index = 0)._name +"-test"+ str(members.index(model)+1)

    out.append(model(commonInput))



  modeltmp = concatenate(out,axis=-1)

  modeltmp = Dense(32, activation='relu')(modeltmp)

  modeltmp = Dense(16, activation='relu')(modeltmp)

  modeltmp = Dense(n_classes, activation='softmax')(modeltmp)

  stacked_model = Model(commonInput,modeltmp)

  stacked_model.compile( loss='categorical_crossentropy',optimizer= 'adam', metrics=['accuracy'])



  return stacked_model
#members=[model,model_2,model_3]

members=[model,model_2,model_3]
stacked_model= stacking_ensemble(members,(img_h,img_w,3),8)

stacked_model.summary()
stacked_model.fit(train_datagen.flow(x_train, to_categorical(y_train,8), batch_size=32),

                    epochs=10,

          callbacks=callbacks,

          validation_data= valid_datagen.flow(x_valid, to_categorical(y_valid,8), batch_size=32),

          verbose=1

             )
labels = stacked_model.predict(test_img)

print(labels[:4])

label = [np.argmax(i) for i in labels]

class_label = [inverse_map[x] for x in label]

print(class_label[:3])

submission = pd.DataFrame({ 'Image': test.Image, 'target': class_label })

submission.head(10)

submission.to_csv('submission.csv', index=False)