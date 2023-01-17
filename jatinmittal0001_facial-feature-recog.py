# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/training/training.csv')
test = pd.read_csv('../input/test/test.csv')
lookid_data = pd.read_csv('../input/IdLookupTable.csv')
train.head().T
# 30 labels(target variables) + 1 column of image pixels data
train_image = train['Image']
y = train.iloc[:,:-1]
test_copy = test.copy()
test = test.drop(['ImageId'],axis=1)
#MISSING VALUE checking
number_missing = y.isnull().sum()
percentage_missing = (100*y.isnull().sum()/y.isnull().count())
missing_data = pd.concat([number_missing,percentage_missing],keys=['number_missing','percentage_missing'],axis=1)
missing_data
# There is almost 68% data missing of some variables, we can't drop the variables nor can remove the
# rows. Either do missing value imputation by mean or previous value or just run model without it and see
# y1 is target variable with missing value imputed
y1 = y.fillna(y.mean(axis=0))                 #score 4.4
y.fillna(method = 'ffill',inplace=True)       # score 3.3
# coverting object type to numeric
# this is good function to convert multiple object type cols to numeric at once
y = y.convert_objects(convert_numeric=True)     
# converting image column data to reqd. numeric format and correct shape
ntrain = train.shape[0]
image = []
for i in range (0,ntrain):
    img = train_image[i].split(' ')
    img = [0 if x =='' else x for x in img]
    image.append(img)

image_list = np.array(image,dtype = 'float')
X = image_list.reshape(-1,96,96,1)
type(X)
# scaling pixel values
X = X/255
ntest = test.shape[0]
image = []
for i in range (0,ntest):
    img = test['Image'][i].split(' ')
    img = [0 if x =='' else x for x in img]
    image.append(img)
image_list = np.array(image,dtype = 'float')
test_data = image_list.reshape(-1,96,96,1)
test_data = test_data/255
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
model = Sequential()

'''
NOTE: here we are not using conv2D layer, bcoz we want to detect particular points at multile location
Conv2D could collapse points data when filter is applied, thereby removing important information
Althogh, I tried Conv2D also and its variations, it was giving similar performance. 
But I beileve not using it would be better
'''
model.add(Flatten(input_shape = (96,96,1)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dense(30))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', metrics=['mae'],optimizer='adam')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# This is used for callbacks, but it is not giving much improvement in model. This could rather lead
# to overfitting for validation set.
history = model.fit(x_train, y_train, batch_size=150, epochs=150, verbose=1,validation_data = (x_test, y_test))
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#predicting test_data
y_pred = model.predict(test_data)
df = pd.DataFrame(y_pred)
df.columns = train.columns[0:30]
df = df.T
df.head()
sub = lookid_data

for i in range(sub.shape[0]):
    row = sub.loc[i,'FeatureName']
    col = sub.loc[i,'ImageId'] - 1
    sub.loc[i,'Location'] = df.loc[row, col]
sub = sub.drop(['ImageId', 'FeatureName'],axis=1)
sub.head()
sub.to_csv('facial_2.csv',index=False)


