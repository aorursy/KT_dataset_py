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





df = pd.read_csv('../input/data.csv')

df.shape





X =  df[['Age','Potential','International Reputation','Weak Foot','Skill Moves','Jersey Number','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys',]]

print(X.shape)

y = df['Overall']

df.columns

X.shape

print(X.head().T)



print(y.shape)



print(y.head())







import keras 

from keras.models import Sequential

model = Sequential()



from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=11))

model.add(Dense(units=1, activation='linear'))



model.compile(loss='mean_squared_error',

              optimizer='sgd')#,

#              metrics=['mean_squared_error'])



#model.compile(loss=keras.losses.mean_squared_error,

#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))



# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) #, stratify = y)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_sts = scaler.fit_transform(X_train)

X_test_sts = scaler.transform(X_test)



model.fit(X_train, y_train, epochs=5, batch_size=32)



loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)





classes = model.predict(X_test, batch_size=128)
classes