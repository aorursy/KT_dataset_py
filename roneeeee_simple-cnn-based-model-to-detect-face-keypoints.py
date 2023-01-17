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
#set directories
train_zip_path = "../input/facial-keypoints-detection/training.zip"
test_zip_path = "../input/facial-keypoints-detection/test.zip"
id_lookup_table = "../input/facial-keypoints-detection/IdLookupTable.csv"
sample_Submission = "../input/facial-keypoints-detection/SampleSubmission.csv"
# unzipping
import zipfile
with zipfile.ZipFile(train_zip_path,'r') as zip_ref:
    zip_ref.extractall('')
with zipfile.ZipFile(test_zip_path,'r') as zip_ref:
    zip_ref.extractall('')

#load datasets
train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('test.csv')
idLookupTable = pd.read_csv(id_lookup_table)
sampleSumission = pd.read_csv(sample_Submission)
train_df.info()
#fill the nan values
train_df.fillna(method='ffill',inplace=True)
# Separate and reshape input values(x_train)
image_df = train_df['Image']
imageArr = []
for i in range(0,len(image_df)):
    img = image_df[i].split()
    img = ['0' if x == '' else x for x in img]
    imageArr.append(img)

x_train = np.array(imageArr,dtype='float')
x_train = x_train.reshape(-1,96,96,1)
print(x_train.shape)
#separate target values (y_train)
keypoints_df = train_df.drop('Image',axis = 1)
y_train = np.array(keypoints_df,dtype='float')
print(y_train.shape)
def visualizeWithNoKeypoints(index):
    plt.imshow(x_train[index].reshape(96,96),cmap='gray')

def visualizeWithKeypoints(index):
    plt.imshow(x_train[index].reshape(96,96),cmap='gray')
    for i in range(1,31,2):
        plt.plot(y_train[0][i-1],y_train[0][i],'ro')
    
#visualizing images with keypoints

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

plt.subplot(1,2,1)
visualizeWithNoKeypoints(1)
plt.subplot(1,2,2)
visualizeWithKeypoints(1)
#import all necessary libraries

from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D,MaxPooling2D,BatchNormalization, Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
model = Sequential()
model.add(Convolution2D(32,(3,3),padding='same',use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32,(3,3),padding='same',use_bias = False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,(3,3),padding='same',use_bias = False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))

model.summary()
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae','acc'])
model.fit(x_train,y_train,batch_size=256,epochs=50,validation_split=2.0)
test_df.isnull().any()
image_df = test_df['Image']
keypoints_df = test_df.drop('Image',axis = 1)
imageArr = []

for i in range(0,len(image_df)):
    img = image_df[i].split()
    img = ['0' if x=='' else x for x in img]
    imageArr.append(img)

x_test = np.array(imageArr,dtype='float')
x_test = x_test.reshape(-1,96,96,1)
print(x_test.shape)
y_test = np.array(keypoints_df,dtype='float')
print(y_test.shape)
#predict our results
pred = model.predict(x_test)
idLookupTable.head()
feature_names = list(idLookupTable['FeatureName'])
image_ids = list(idLookupTable['ImageId']-1)
row_ids = list(idLookupTable['RowId'])

feature_list = []
for feature in feature_names:
    feature_list.append(feature_names.index(feature))
    
predictions = []
for x,y in zip(image_ids, feature_list):
    predictions.append(pred[x][y])
    
row_ids = pd.Series(row_ids, name = 'RowId')
locations = pd.Series(predictions, name = 'Location')
locations = locations.clip(0.0,96.0)
submission_result = pd.concat([row_ids,locations],axis = 1)
submission_result.to_csv('Submission.csv',index = False)
