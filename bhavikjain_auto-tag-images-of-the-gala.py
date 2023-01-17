# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import pandas as pd

import numpy as np

import seaborn as sns

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt
train=pd.read_csv('/kaggle/input/classification-of-images/dataset/train.csv')

test=pd.read_csv('/kaggle/input/classification-of-images/dataset/test.csv')

train.head()
Class_map={'Food':0,'Attire':1,'Decorationandsignage':2,'misc':3}

inverse_map={0:'Food',1:'Attire',2:'Decorationandsignage',3:'misc'}

train['Class']=train['Class'].map(Class_map)
train_img=[]

train_label=[]

j=0

path='/kaggle/input/classification-of-images/dataset/Train Images'

for i in tqdm(train['Image']):

    final_path=os.path.join(path,i)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(224,224))

    img=img.astype('float32')

    train_img.append(img)

    train_label.append(train['Class'][j])

    j=j+1
test_img=[]

path='/kaggle/input/classification-of-images/dataset/Test Images'

for i in tqdm(test['Image']):

    final_path=os.path.join(path,i)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(224,224))

    img=img.astype('float32')

    test_img.append(img)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagenerator = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        rotation_range=10,  

        zoom_range = 0.20,  

        width_shift_range=0.20,  

        height_shift_range=0.20,  

        horizontal_flip=True,  

        vertical_flip=False) 





datagenerator.fit(train_img)
train_img=np.array(train_img)

test_img=np.array(test_img)

train_label=np.array(train_label)

print(train_img.shape)

print(test_img.shape)

print(train_label.shape)
from tensorflow.keras.applications.vgg16 import VGG16



from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization



from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers import *

from keras.models import Sequential

from keras.applications.resnet50 import ResNet50



base_model = ResNet50(

    weights='imagenet',

    include_top=False, 

    input_shape=(224, 224, 3), 

    pooling='avg',

)

base_model.trainable = False



model = Sequential([

  base_model,

  Dense(4, activation='softmax'),

])
from keras.applications import VGG19

from keras.applications.vgg19 import preprocess_input

from keras.layers import *

from keras.models import Sequential

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = (224, 224, 3),pooling='avg')

vgg19.trainable = False



model = Sequential([

  vgg19, 

  Dense(1024, activation='relu'),

  Dropout(0.5),

  Dense(256, activation='relu'),

  Dense(4, activation='softmax'),

])
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop



vgg19.trainable=False



reduce_learning_rate = ReduceLROnPlateau(monitor='loss',

                                         factor=0.1,

                                         patience=2,

                                         cooldown=2,

                                         min_lr=0.00001,

                                         verbose=1)



callbacks = [reduce_learning_rate]

    





model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(datagenerator.flow(train_img, to_categorical(train_label,4), batch_size=64),

                    epochs=20,callbacks=callbacks)

labels = model.predict(test_img)

print(labels[:4])

label = [np.argmax(i) for i in labels]

class_label = [inverse_map[x] for x in label]

print(class_label[:3])

submission = pd.DataFrame({ 'Image': test.Image, 'Class': class_label })

submission.head(10)

submission.to_csv('result.csv', index=False)