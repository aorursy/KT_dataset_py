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
df = pd.read_csv('/kaggle/input/framingham-heart-study-dataset/framingham.csv')
df
from keras.models import Sequential

from keras.layers import Dense, Dropout,SimpleRNNCell

from keras import optimizers

from keras import layers

model = Sequential()

model.add(Dense(units=32,activation='relu', input_dim= 15))##

model.add(Dropout(0.05))

model.add(Dense(units=64, activation='relu'))##, input_dim=300))

model.add(Dropout(0.1))

model.add(Dense(units=128, activation='relu'))##, input_dim=500))

model.add(Dropout(0.2))

model.add(Dense(units=364,activation='relu'))

model.add(SimpleRNNCell(units=364,activation='relu'))##, input_dim= 1000))

model.add(Dropout(0.3))

model.add(Dense(units=128,activation='relu'))##, input_dim= 500))

model.add(Dropout(0.2))

model.add(Dense(units=64,activation='relu'))##, input_dim= 30))

model.add(Dropout(0.1))

model.add(Dense(units=32,activation='relu'))##, input_dim= 30))

model.add(Dense(units=16,activation='relu'))##, input_dim= 30))

model.add(Dense(units=8,activation='relu'))##, input_dim= 30))



model.add(Dense(units=1, activation='sigmoid'))

sgd = optimizers.Adam(lr=0.001)

model.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
y = df["TenYearCHD"]

##y


X = df[df.columns[:-1]]



##X
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
transformed_values = imputer.fit_transform(X)

##transformed_values
for j in range(15):

    min_x = min(transformed_values[:,j])

    max_x = max(transformed_values[:,j])

    for i in range(len(transformed_values[:,j])):

        

        transformed_values[:,j][i] = (transformed_values[:,j][i] - min_x) / (max_x - min_x)

##transformed_values
from sklearn.model_selection import train_test_split

X_train,x_test,y_train,y_test = train_test_split(transformed_values,y, test_size = 0.10, random_state = 42, shuffle = True)
model.fit(X_train,y_train, epochs = 100,shuffle=True)
_,acc = model.evaluate(x_test,y_test)

print("Acc:{}",acc*100)