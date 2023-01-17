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
df = pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
df.head()
df['Gender'].replace({'Male':0,'Female':1},inplace=True)
df.head()
# Without Normalising



X = df.iloc[:,1:-1].values

y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=0)
import tensorflow 

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
model = Sequential()

model.add(Dense(10,input_dim=3,activation='relu'))

model.add(Dense(10,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=2000,validation_data=(X_test,y_test),verbose=1)
import matplotlib.pyplot as plt



plt.plot(history.history['val_accuracy'])
# Normalize inputs



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_trains,X_tests,y_trains,y_tests = train_test_split(X_scaled,y,test_size=0.2,random_state=1)
model1 = Sequential()

model1.add(Dense(10,input_dim=3,activation='relu'))

model1.add(Dense(10,activation='relu'))

model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model1.fit(X_trains,y_trains,epochs=200,validation_data=(X_tests,y_tests),verbose=1)
plt.plot(history.history['val_accuracy'])
X_scaled