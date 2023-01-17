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
import seaborn as sns
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
train=pd.read_csv('/kaggle/input/identify-the-dance-form/train.csv')
test=pd.read_csv('/kaggle/input/identify-the-dance-form/test.csv')
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
from keras.utils import to_categorical
train_img=[]
train_label=[]
j=0
path='/kaggle/input/identify-the-dance-form/train'
for i in tqdm(train['Image']):
    final_path=os.path.join(path,i)
    img=cv2.imread(final_path)
    img=cv2.resize(img,(img_h,img_w))
    img=img.astype('float32')
    train_img.append(img)
    train_label.append(train['target'][j])
    j=j+1
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_img,train_label, test_size=0.3, shuffle= True)
test_img=[]
path='/kaggle/input/identify-the-dance-form/test'
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
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

test_datagen= ImageDataGenerator(rescale=1./255)
valid_datagen= ImageDataGenerator(rescale=1./255)
train_datagen.fit(x_train)
test_datagen.fit(test_img)
valid_datagen.fit(x_valid)

from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
base_model_3=VGG19(include_top=False, weights='imagenet',input_shape=(img_h,img_w,3), pooling='max')

res_name = []
for layer in base_model_3.layers:
    res_name.append(layer.name)
    
set_trainable = False
for layer in base_model_3.layers:
    if layer.name in res_name[:-4]:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
model_3=Sequential()
model_3.add(base_model_3)
model_3.add(Flatten())

model_3.add(Dense(2048, activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dropout(0.3))


model_3.add(Dense(1024, activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dense(512, activation='relu'))

model_3.add(Dense(256, activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dropout(0.3))
model_3.add(Dense(128, activation='relu'))
model_3.add(Dense(8,activation='softmax'))

model_3.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_3.summary()
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
model_3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_3.fit(train_datagen.flow(x_train, to_categorical(y_train,8), batch_size=32),epochs=40,
          validation_data= valid_datagen.flow(x_valid, to_categorical(y_valid,8), batch_size=32),
          verbose=1)
labels = model_3.predict(test_img)
print(labels[:4])
label = [np.argmax(i) for i in labels]
class_label = [inverse_map[x] for x in label]
print(class_label[:3])
submission = pd.DataFrame({ 'Image': test.Image, 'target': class_label })
submission.head(10)
submission.to_csv('submission.csv', index=False)







