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
import pandas as pd



import matplotlib.pyplot as plt



import numpy as np

 

import seaborn as sns
data=pd.read_csv("../input/car-purchase-data/Car_Purchasing_Data.csv",encoding="latin-1")

data
sns.pairplot(data)
X=data.drop(['Car Purchase Amount','Country','Customer Name','Customer e-mail'],axis=1)

X
y=data['Car Purchase Amount']

y
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_scale=scaler.fit_transform(X)

X_scale
y=y.values.reshape(-1,1)
y_scale=scaler.fit_transform(y)
y_scale
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_scale,y_scale,test_size=0.15)



X_test.shape
X_train.shape
import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense



model=Sequential()

model.add(Dense(40,input_dim=5,activation='relu'))

model.add(Dense(40,activation='relu'))

model.add(Dense(1,activation='linear'))
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')
epochs_hist=model.fit(X_train,Y_train,epochs=100,batch_size=25,verbose=1,validation_split=0.2)
# Gender,Age,Annual Salary,Credit card debt,Net Worth 

X_test=np.array([[1,50,50000,10000,600000]])

y_predict=model.predict(X_test)

print(y_predict)