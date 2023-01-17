# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import keras
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
train['image_id']=train['image_id']+'.jpg'
test['image_id']=test['image_id']+'.jpg'
train.head()
img_size=256
train_images=[]
pro_tr_images=[]
filename=train.image_id
for file in filename:
    image_tr=cv2.imread("../input/plant-pathology-2020-fgvc7/images/"+file)
    res_tr=cv2.resize(image_tr,(img_size,img_size))
    train_images.append(res_tr)
    #image_tr = cv2.cvtColor(image_tr, cv2.COLOR_BGR2GRAY)
    #pro_tr_images.append(image_tr)
    
train_images=np.array(train_images)
#pro_tr_images=np.array(pro_tr_images)
pro_tr_images=np.array(pro_tr_images)
pro_tr_images[0].shape
test_images=[]
pro_test_images=[]
filename=test.image_id
for file in filename:
    image=cv2.imread("../input/plant-pathology-2020-fgvc7/images/"+file)
    res=cv2.resize(image,(img_size,img_size))
    test_images.append(res)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #pro_test_images.append(image)
    
test_images=np.array(test_images)
#pro_test_images=np.array(pro_test_images)
from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()
for i in range(len(pro_tr_images)):
    #pro_tr_images[i] = mm.fit_transform(pro_tr_images[i])
    train_images= mm.fit_transform(train_images)
    
for j in range(len(pro_test_images)):
    #pro_test_images[j] = mm.fit_transform(pro_test_images[j])
    test_images= mm.fit_transform(test_images)
plt.figure(figsize=(15,15))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(pro_tr_images[i])
train_labels = np.float32(train.loc[:, 'healthy':'scab'].values)
"""
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    fill_mode='nearest',
    shear_range=0.1,
    rescale=1/255,
    brightness_range=[0.5, 1.5])
"""
from keras.utils import plot_model
from keras.layers import Flatten
import math
from keras.models import Sequential
from keras.layers import LSTM,Conv2D,MaxPooling2D,Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
def cnn_model(train_X, train_Y,test_X,epochs):
    input_tensor = Input(shape=(img_size, img_size,3))
    base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
    x = base_model.output
    
    #model.add(Conv2D(filters=32, kernel_size=5,padding='same',input_shape=(img_size, img_size, 3)))
    
    #x = MaxPooling2D()(x)
    x = Flatten()(x)
    #x = Dense(8000, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    
    plot_model(model)
    history = model.fit(train_X, train_Y, validation_split=0.1, epochs=epochs)
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend() 
    plt.show()
    result = model.predict(test_X)
    
    return result
from sklearn.preprocessing import MinMaxScaler
epochs = 100
result = cnn_model(train_images, train_labels,test_images,epochs)
cl = pd.DataFrame(result)
cl.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
df_end = pd.concat([submission, cl],axis=1)
df_end.to_csv('submission.csv', index=False)
df_end