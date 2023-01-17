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
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from keras.models import Sequential

from keras.layers import Dense, Dropout

import keras

import tensorflow as tf
data = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')

X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values
label_enc_X = LabelEncoder()

X[:,1] = label_enc_X.fit_transform(X[:,1])



label_enc_y = LabelEncoder()

y = label_enc_y.fit_transform(y)
standard_scalar = StandardScaler()

X = standard_scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
model = Sequential()



model.add(Dense(32, input_dim = 10, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.add(Dense(64, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.add(Dense(64, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 64, epochs = 500, verbose = 1)
score = model.evaluate(X_test, y_test)

print(score)