import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as p

import PIL as pil

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import seaborn as sns
base='/kaggle/input/skin-cancer-mnist-ham10000'
meta=pd.read_csv(os.path.join(base,'HAM10000_metadata.csv'))

meta.info()
meta.head()
g = sns.catplot(x="dx", kind="count", palette='bright', data=meta)

g.fig.set_size_inches(16, 5)



g.ax.set_title('Skin Cancer by Class', fontsize=20)

g.set_xlabels('Skin Cancer Class', fontsize=14)

g.set_ylabels('Number of Data Points', fontsize=14)
g = sns.catplot(x="dx", kind="count", hue="sex", palette='coolwarm', data=meta)

g.fig.set_size_inches(16, 5)



g.ax.set_title('Skin Cancer by Sex', fontsize=20)

g.set_xlabels('Skin Cancer Class', fontsize=14)

g.set_ylabels('Number of Data Points', fontsize=14)

g._legend.set_title('Sex')
g = sns.catplot(x="dx", kind="count", hue="age", palette='bright', data=meta)

g.fig.set_size_inches(16, 9)



g.ax.set_title('Skin Cancer by Age', fontsize=20)

g.set_xlabels('Skin Cancer Class', fontsize=14)

g.set_ylabels('Number of Data Points', fontsize=14)

g._legend.set_title('Age')
df=pd.read_csv(os.path.join(base,'hmnist_28_28_RGB.csv'))

x=df.drop('label',axis=1)

y=df['label']

x=x.to_numpy()

x=x/255

y=to_categorical(y)
df['label'].value_counts()
label={

    ' Actinic keratoses':0,

    'Basal cell carcinoma':1,

    'Benign keratosis-like lesions':2,

    'Dermatofibroma':3,

    'Melanocytic nevi':4,

    'Melanoma':6,

    'Vascular lesions':5

}
x=x.reshape(-1,28,28,3)
p.figure(figsize=(20,10))

for i in range(25):

    p.subplot(5,5,i+1)

    img=x[i]

    p.imshow(img)
trainx,trainy,testx,testy = train_test_split(x,y,test_size=0.02,random_state=13)
datagen=ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               width_shift_range=0.10, # Shift the pic width by a max of 5%

                               height_shift_range=0.10, # Shift the pic height by a max of 5%

                               rescale=1/255, # Rescale the image by normalzing it.

                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)

                               zoom_range=0.1, # Zoom in by 10% max

                               horizontal_flip=True,

                               vertical_flip=True,

                               fill_mode='nearest')

datagen1=ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               width_shift_range=0.10, # Shift the pic width by a max of 5%

                               height_shift_range=0.10, # Shift the pic height by a max of 5%

                               rescale=1/255, # Rescale the image by normalzing it.

                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)

                               zoom_range=0.1, # Zoom in by 10% max

                               horizontal_flip=True,

                               vertical_flip=True,

                               fill_mode='nearest')
datagen.fit(trainx)

datagen1.fit(trainy)
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D

from tensorflow.keras.models import Sequential

from tensorflow.keras.metrics import Recall

from tensorflow.keras.optimizers import RMSprop



model=Sequential()



model.add(Conv2D(64,(2,2),input_shape=(28,28,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())



model.add(Conv2D(512,(2,2),input_shape=(28,28,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())



model.add(Dropout(0.3))



model.add(Conv2D(1024,(2,2),input_shape=(28,28,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())



model.add(Dropout(0.4))



model.add(Conv2D(1024,(1,1),input_shape=(28,28,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(BatchNormalization())



model.add(Dropout(0.4))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))





model.add(Dense(7,activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',Recall()])

model.summary()
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau



early=EarlyStopping(monitor='accuracy',patience=3)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.0001)
model.fit(trainx,testx,batch_size=90,epochs=25,validation_data=(trainy,testy),callbacks=[reduce_lr,early])
p.figure(figsize=(15,5))

loss=pd.DataFrame(model.history.history)

loss=loss[['accuracy','val_accuracy']]

loss.plot()
from sklearn.metrics import classification_report,confusion_matrix



predictions=model.predict_classes(trainy)



check=[]

for i in range(len(testy)):

  for j in range(7):

    if(testy[i][j]==1):

      check.append(j)

check=np.asarray(check)



print(classification_report(check,predictions))
