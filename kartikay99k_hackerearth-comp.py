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
train['Class']
train_img=[]

train_label=[]

j=0

path='/kaggle/input/classification-of-images/dataset/Train Images'

for i in tqdm(train['Image']):

    final_path=os.path.join(path,i)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(150,150))

    img=img.astype('float32')

    train_img.append(img)

    train_label.append(train['Class'][j])

    j=j+1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.20)

datagen.fit(train_img)



test_img=[]

path='/kaggle/input/classification-of-images/dataset/Test Images'

for i in tqdm(test['Image']):

    final_path=os.path.join(path,i)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(150,150))

    img=img.astype('float32')

    test_img.append(img)
train_img=np.array(train_img)

test_img=np.array(test_img)

train_label=np.array(train_label)

print(train_img.shape)

print(test_img.shape)

print(train_label.shape)
from tensorflow.keras.applications.inception_v3 import InceptionV3



from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import GlobalAveragePooling2D



from tensorflow.keras.callbacks import ReduceLROnPlateau

base_model=InceptionV3(input_shape = (150, 150, 3), 

                                include_top = False, 

                                weights = 'imagenet')

model=Sequential()



model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (150,150,3))) 

model.add(BatchNormalization())



model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (150,150,3))) 

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu')) 

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu')) 

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu')) 

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu')) 

model.add(BatchNormalization())

model.add(MaxPooling2D())



model.add(GlobalAveragePooling2D())



model.add(Dense(256,activation='relu'))

model.add(Dropout(0.3))



model.add(Dense(4,activation='softmax'))



model.summary()

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop





reduce_learning_rate = ReduceLROnPlateau(monitor='loss',

                                         factor=0.1,

                                         patience=2,

                                         cooldown=2,

                                         min_lr=0.00001,

                                         verbose=1)



callbacks = [reduce_learning_rate]

    





model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(datagen.flow(train_img, to_categorical(train_label,4), batch_size=32),

                    epochs=120,callbacks=callbacks)
labels = model.predict(test_img)

print(labels[:4])

label = [np.argmax(i) for i in labels]

class_label = [inverse_map[x] for x in label]

print(class_label[:3])

submission = pd.DataFrame({ 'Image': test.Image, 'Class': class_label })

submission.head(10)

submission.to_csv('submission.csv', index=False)