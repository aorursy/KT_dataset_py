# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.info()
data.describe()
data.target.value_counts()
data.tail()
from sklearn.model_selection import train_test_split
data.head(5)
X = data.drop(['ID_code','target'],axis = 1)
X.shape
y = data['target']
y.shape
### Steps to improve the performance
len(y[y==0].index)
### Normalize the data first
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
### Undersample values with y == 0
print(X.shape)
print(X.shape[0])
n_train = X.shape[0]

print(n_train)

no_samples = int(0.6 * n_train)
no_samples
drop_indices = np.random.choice(y[y == 0].index,no_samples,replace = False)
len(drop_indices)
print('Shape b/f Undersampling')

print('X: {}, y:{}'.format(X.shape,y.shape))
X = X.drop(drop_indices,axis = 0)

y = y.drop(drop_indices,axis = 0)
print('Shape after undersampling')

print('X: {},y:{}'.format(X.shape,y.shape))
len_y_0 = len(y[y==0].index)
len_y_0
len_y_1 = len(y[y==1].index)
len_y_1
y = pd.Series(y,name = 'target')
y.value_counts()
data = pd.concat([X,y],axis = 1).reset_index(drop = True)
data.target.value_counts()
dataset = data.sample(frac = 1).reset_index(drop = True)
X = dataset.drop('target',axis = 1)
y = dataset['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from tensorflow.python.keras.layers import Dense,Dropout,BatchNormalization

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.callbacks import EarlyStopping

from tensorflow.contrib.layers import l2_regularizer

from tensorflow.python.keras.optimizers import SGD
model = Sequential()

#model.add(BatchNormalization(input_shape = (201)))

model.add(Dense(units = 2048,kernel_initializer='truncated_normal',activation='relu'))

model.add(Dropout(rate=0.5))

model.add(BatchNormalization())

model.add(Dense(units = 1024,activation = 'relu'))

model.add(Dropout(rate = 0.5))

model.add(BatchNormalization())

model.add(Dense(units = 128,activation = 'relu'))

model.add(Dense(units = 8,activation = 'relu'))

model.add(Dense(units = 1,activation = 'sigmoid'))
optimizer = SGD(lr = 0.01)
callback = EarlyStopping(patience = 2)
model.compile(SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),loss = 'binary_crossentropy',metrics = ['accuracy'])
X_train_nn = np.array(X_train)

y_train_nn = np.array(y_train)
model.fit(X_train_nn,y_train_nn,epochs = 10,validation_split = 0.05,batch_size = 32,callbacks = [callback],shuffle = True)
data_sub = pd.read_csv('../input/test.csv')
ids = data_sub['ID_code']
features = data_sub.drop('ID_code',axis = 1)
predictions = model.predict(features)
pred_nn = np.array(predictions)

for i in range(len(pred_nn)):

    if pred_nn[i] >= 0.5:

        pred_nn[i] = 1

    else:

        pred_nn[i] = 0

pred_nn = np.array(pred_nn,dtype = np.int32)
sub = pd.DataFrame()

sub['ID_code'] = ids

sub['target'] = pred_nn

sub.head()

sub.set_index('ID_code')

sub['target'].value_counts()

sub.to_csv('submissions.csv',index = False)