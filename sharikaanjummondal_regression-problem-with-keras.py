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
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df.shape
df.sample(3)
df.isnull().sum()
df.drop(columns=['Serial No.'],inplace=True)
df.head()
X=df.iloc[:,:-1].values

y=df.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
import tensorflow

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(7,activation='relu',input_dim=X_train.shape[1]))

model.add(Dense(7,activation='relu'))

model.add(Dense(1,activation='linear'))
model.summary()
model.compile(optimizer='Adam',loss='mean_squared_error')
history = model.fit(X_train,y_train,epochs=100,batch_size=10,verbose=1,validation_split=0.2)
y_pred=model.predict(X_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
history.history



import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])