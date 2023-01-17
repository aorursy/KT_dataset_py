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
train = pd.read_csv('/kaggle/input/hmif-data-science-bootcamp-2019/train-data.csv')

test = pd.read_csv('/kaggle/input/hmif-data-science-bootcamp-2019/test-data.csv')
test.head()
train.columns
train.dtypes
train.head()
test.head()
train['bentuk'] = train['bentuk'].astype('category').cat.codes

test['bentuk'] = test['bentuk'].astype('category').cat.codes

train['status'] = train['status'].astype('category').cat.codes

test['status'] = test['status'].astype('category').cat.codes

train['kurikulum'] = train['kurikulum'].astype('category').cat.codes

test['kurikulum'] = test['kurikulum'].astype('category').cat.codes

train['penyelenggaraan'] = train['penyelenggaraan'].astype('category').cat.codes

test['penyelenggaraan'] = test['penyelenggaraan'].astype('category').cat.codes

train['akses_internet'] = train['akses_internet'].astype('category').cat.codes

test['akses_internet'] = test['akses_internet'].astype('category').cat.codes

train['sumber_listrik'] = train['sumber_listrik'].astype('category').cat.codes

test['sumber_listrik'] = test['sumber_listrik'].astype('category').cat.codes
train.head()
test.head()
train.shape
test.shape
X_train = train.iloc[:,:46].values

y_train = train.iloc[:,46:47].values



X_test = test.iloc[:,:].values
X_train.shape
y_train.shape
y_train
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder()

y_train = ohe.fit_transform(y_train).toarray()
y_train
import keras

from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(64, input_dim=46, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=64)
y_pred = model.predict(X_test)

#Converting predictions to label

pred = list()

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))
pred
sub = pd.read_csv('/kaggle/input/hmif-data-science-bootcamp-2019/sample-submission.csv')
sub['akreditasi'] = pred
sub.to_csv('sub.csv', index = False)