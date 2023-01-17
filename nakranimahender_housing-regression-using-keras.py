# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import matplotlib.pyplot as plt



import tensorflow as tf

import keras

from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from keras.layers import Dense, Activation

from keras.utils import np_utils 

from keras import optimizers 

from keras.models import load_model

from keras.models import model_from_json



import keras.backend as K

import xgboost as xgb

from sklearn.metrics import mean_squared_error





# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

print(train_data.shape)
Y_train = train_data["SalePrice"]

print(Y_train.shape)
test_data = pd.read_csv("../input/test.csv")

print(test_data.shape)

#test_data.head()
list2 = ["LotArea","OverallQual","YearBuilt","YearRemodAdd","BsmtFinSF1","TotalBsmtSF","1stFlrSF","2ndFlrSF",

         "GrLivArea","FullBath","HalfBath", "TotRmsAbvGrd","Fireplaces","GarageCars", "GarageArea",

         "WoodDeckSF","OpenPorchSF"]

for col in train_data.columns:

    if col not in list2:

        train_data = train_data.drop(col, axis=1)

for col in test_data.columns:

    if col not in list2:

        test_data = test_data.drop(col, axis=1)

        

print(train_data.shape)

print(test_data.shape)
for col in train_data.columns:

    nan_rows = train_data[train_data[col].isnull()]

    index = list(nan_rows.index)

    for i in index:

        if train_data[col].dtype == 'object':

            train_data[col][i] = 'None'

        else:

            train_data[col][i] = 0

            

train_data.head()
for col in test_data.columns:

    nan_rows = test_data[test_data[col].isnull()]

    index = list(nan_rows.index)

    for i in index:

        if test_data[col].dtype == 'object':

            test_data[col][i] = 'None'

        else:

            test_data[col][i] = 0

            

test_data.head()
whole_data1 = pd.concat([train_data,test_data])

print(whole_data1.shape)

whole_data1.head()
for i in whole_data1.columns:

    if whole_data1[i].dtype == 'object':

        le = preprocessing.LabelEncoder()

        le.fit(whole_data1[i])

        train_data[i]=le.transform(train_data[i])

        test_data[i]=le.transform(test_data[i])





print(train_data.shape)

#train_data.head()
list2 = ["LotArea","BsmtFinSF1","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea", "GarageArea","WoodDeckSF",

         "OpenPorchSF"]



for col in train_data.columns:

    if col in list2:

        train_data[col] = np.log1p(train_data[col])

        

for col in test_data.columns:

    if col in list2:

        test_data[col] = np.log1p(test_data[col])

        

train_data.head()
Y_train = np.log1p(Y_train)

Y_train.head()

x_train, x_val, y_train, y_val = train_test_split(train_data, Y_train, test_size = 0.15, random_state = 2)

print("x_train shape: ",x_train.shape)

print("x_val shape: ",x_val.shape)

print("y_train shape: ",y_train.shape)

print("y_val shape :",y_val.shape)
K.clear_session()
def root_mean_squared_error(y_true, y_pred):

        return (K.sqrt(K.mean(K.square(y_pred - y_true))))
def regressionmodel():

    model=Sequential()

    model.add(Dense(17, input_dim=17,activation='relu' ))

    model.add(keras.layers.BatchNormalization())

    model.add(Dense(8,activation='relu'))

    model.add(Dense(1))

    model.compile(loss=root_mean_squared_error, optimizer='adam')

    return model
reg1 = KerasRegressor(build_fn=regressionmodel, batch_size=32,epochs=256)
reg1.fit(x_train, y_train, verbose=2, validation_data=(x_val, y_val))
pred = reg1.predict(test_data, batch_size=32, verbose=1)
y_classes=[]

for i in pred:

    y_classes.append(np.expm1(i))

    

print("predict complete")



submissions=pd.DataFrame({"Id": list(range(1461,len(pred)+1461)),

                         "SalePrice": y_classes})

submissions.to_csv("result.csv", index=False, header=True)
Out_data = pd.read_csv("result.csv")

print(Out_data.shape)

Out_data.head()