# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
traindf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
testdf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
submissiondf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
traindf.head()
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.countplot(traindf['healthy'])
sns.countplot(traindf['rust'])
sns.countplot(traindf['multiple_diseases'])
sns.countplot(traindf['scab'])
testdf.head()
# Loading train images 
import os
import cv2
import glob
img_size = 224
path = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

testimages = []
trainimages = []

for img in traindf['image_id']:
    imgpath = os.path.join(path,img) + ".jpg" 
    IMAGE = cv2.imread(imgpath)
    IMAGE=cv2.resize(IMAGE,(img_size,img_size),interpolation=cv2.INTER_AREA)
    trainimages.append(IMAGE)
    
for img in testdf['image_id']:
    imgpath = os.path.join(path,img) + ".jpg"
    IMAGE = cv2.imread(imgpath)
    IMAGE=cv2.resize(IMAGE,(img_size,img_size),interpolation=cv2.INTER_AREA)
    testimages.append(IMAGE)   

len(trainimages) , len(testimages)
fig,ax = plt.subplots(1,4,figsize=(15,15))
for i in range(4):
    ax[i].imshow(trainimages[i])
fig,ax = plt.subplots(1,4,figsize=(15,15))
for i in range(4):
    ax[i].imshow(testimages[i])
# creating X and Y data for training

from keras.preprocessing.image import img_to_array

X = np.ndarray(shape=(len(trainimages),img_size,img_size,3),dtype = np.float32)
i = 0
for img in trainimages:
    X[i] = img_to_array(img)
    X[i] = trainimages[i]
    i += 1
X = X/255.0
y = traindf.drop(columns=['image_id']) # take rest 4 columns
y = np.array(y.values)
y
X.shape,y.shape
# similary for final testing data
X_for_testing = np.ndarray(shape=(len(testimages),img_size,img_size,3),dtype = np.float32)
i = 0
for img in testimages:
    X_for_testing[i] = img_to_array(img)
    X_for_testing[i] = testimages[i]
    i += 1
X_for_testing = X_for_testing/255.0

X_for_testing.shape
# DATA SPLITSSSSS
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
# HANDLING UNEQUAL DATASET USING SMOTE 

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train,y_train = smote.fit_resample(X_train.reshape((-1,img_size*img_size*3)),y_train)

X_train = X_train.reshape((-1,img_size,img_size,3))

X_train.shape,y_train.shape,y_train.sum(axis=0)
print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow import keras 
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam, Adadelta, Adamax
from tensorflow.keras.layers import Dropout , BatchNormalization , Flatten , MaxPool2D,MaxPooling2D , Activation , Dense , Conv2D , InputLayer
datagen = ImageDataGenerator(rotation_range=45,
                             shear_range=.25,
                              zoom_range=.25,
                              width_shift_range=.25,
                              height_shift_range=.25,
                              rescale=1/255,
                              brightness_range=[.5,1.5],
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest')
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

IMAGE_SIZE = [224, 224]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
  layer.trainable = False
x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
prediction = Dense(4, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

lr=ReduceLROnPlateau(monitor='val_accuracy',factor=.5,patience=10,min_lr=.000001,verbose=1)
es=EarlyStopping(monitor='val_loss', patience=20)
callbacks = [lr,es]
batch_size = 32
epochs = 200

vgghistory = model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs=epochs,
                                callbacks=callbacks,
                                steps_per_epoch = X_train.shape[0]//batch_size,
                                verbose=1,
                                validation_data = datagen.flow(X_val,y_val,batch_size=batch_size),
                                validation_steps = X_val.shape[0]//batch_size)

img_size=224
reg = 0.0005

model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5),activation='relu', input_shape=(img_size, img_size, 3), kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Conv2D(128, kernel_size=(5,5),activation='relu', kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))
model.add(Dropout(.25))

model.add(Conv2D(32, kernel_size=(3,3),activation='relu', kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu',kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))
model.add(Dropout(.25))

model.add(Conv2D(128, kernel_size=(5,5),activation='relu', kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Conv2D(512, kernel_size=(5,5),activation='relu',kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))
model.add(Dropout(.25))

model.add(Conv2D(128, kernel_size=(3,3),activation='relu',kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu',kernel_regularizer=l2(reg)))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(MaxPooling2D(pool_size=(2,2), padding='SAME'))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(300,activation='relu'))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Dropout(.25))
model.add(Dense(200,activation='relu'))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Dropout(.25))
model.add(Dense(100,activation='relu'))
model.add(BatchNormalization(axis=-1,center=True,scale=False))
model.add(Dropout(.25))
model.add(Dense(4,activation='softmax'))

model.summary()
batch_size = 32
epochs = 200

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs=epochs,
                                callbacks=callbacks,
                                steps_per_epoch = X_train.shape[0]//batch_size,
                                verbose=1,
                                validation_data = datagen.flow(X_val,y_val,batch_size=batch_size),
                                validation_steps = X_val.shape[0]//batch_size)
import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
pred = model.predict(X_for_testing).argmax(axis=0)
pred
testids = testdf['image_id']
pred = model.predict(X_for_testing)
print(pred)
res = pd.DataFrame()
res['image_id'] = testids
res['healthy'] = pred[:, 0]
res['multiple_diseases'] = pred[:, 1]
res['rust'] = pred[:, 2]
res['scab'] = pred[:, 3]
res.to_csv('submission.csv', index=False)
res.head(10)
