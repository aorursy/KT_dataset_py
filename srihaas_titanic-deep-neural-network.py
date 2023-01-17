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
read = pd.read_csv("../input/train.csv")
#print(read.head())
#print(read.describe())
print(read.columns)
train_y = read.Survived
train_X = read.drop(['Survived', 'Cabin', 'Ticket','Name'], axis=1)
print(train_y.head())
print(train_X.head())
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

train_X = pd.get_dummies(train_X)

my_Imputer = Imputer()
train_X = my_Imputer.fit_transform(train_X)

train_X = preprocessing.normalize(train_X, axis=0)


train_y = train_y.values
train_y = train_y.reshape((891,1))
print(train_y.shape)
print(train_X.shape)
model = Sequential()
model.add(Dense(15, input_dim=11, activation= 'relu'))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(train_X, train_y, validation_split= 0.2, epochs = 147)
test = pd.read_csv("../input/test.csv")
test = test.drop(['Cabin', 'Ticket','Name'], axis=1)

test_i = pd.get_dummies(test)
test_i = my_Imputer.fit_transform(test_i)
test_i = preprocessing.normalize(test_i, axis=0)
print(test_i.shape)
k = model.predict(test_i)



ans = np.zeros((k.shape))
ans = ( k > 0.5)
ans = ans.astype(int)

df = pd.DataFrame(ans, columns = ['Survived'])
pg = test.iloc[:, 0:1]

df = pg.join(df)
print(df.isna().any())
df.to_csv('my_first.csv', index = False)
gen = pd.read_csv("../input/gender_submission.csv")
print(gen.columns == df.columns)

