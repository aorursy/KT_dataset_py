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
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense,LSTM,Dropout

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
data = pd.read_csv("../input/covid19-kerala/Covid19_Kerala.csv")

data = data.drop('Sno',axis=1)

data = data.drop('Date',axis=1)

data = data.drop('Deaths',axis=1)
data.head()
total_conf = data['ConfirmedForeignNational']+data['ConfirmedIndianNational']-data['Cured']

data["As_Of_Today"] = total_conf
X = data.iloc[:,3:7].values

y = data.iloc[:,7].values
normalizer = StandardScaler()

X_norm = normalizer.fit_transform(X)

y = y.reshape(-1,1)

y_norm = normalizer.fit_transform(y)
pre_X = []

pre_y = []

for i in range(10 ,len(y)):

    pre_X.append(X_norm[i-10:i,:])

    pre_y.append(y_norm[i,0])



pre_X = np.array(pre_X)

pre_y = np.array(pre_y)

print(pre_X.shape)

print(pre_y.shape)
X_train,X_test,y_train,y_test = train_test_split(pre_X,pre_y,test_size=0.2,random_state=4)
model = Sequential()

model.add(LSTM(units = 100,return_sequences=True,input_shape = (pre_X.shape[1],4)))

model.add(LSTM(units = 350,return_sequences=True))

model.add(LSTM(units = 350,return_sequences=True))

model.add(LSTM(units = 100))

model.add(Dense(units=1,activation='linear'))

model.compile(optimizer='adam',loss = 'mean_squared_error',metrics = ['mean_absolute_error'])



res = model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test),batch_size=4)
results = model.predict(X_test)

plt.scatter(range(10),results,c='r')

plt.scatter(range(10),y_test,c='g')

plt.show()
plt.plot(res.history['loss'])

plt.show()
result = model.predict(pre_X)

plt.scatter(range(49),result,c='r')

plt.scatter(range(49),pre_y,c='g')

plt.show()
from tensorflow.keras.models import save_model

save_model(model,"covid_predictor.h5")
future_X = np.array([X_norm[-10:len(X_norm),:]])

future_y = np.array(y_norm[-10:len(X_norm),0])

future_X.shape
res = model.predict(future_X)

print(normalizer.inverse_transform(res))