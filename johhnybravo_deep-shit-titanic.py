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
train = pd.read_csv("../input/train.csv").fillna(0)
train
X = np.array(train.iloc[:, [2, 4, 5, 6, 7, 9, 10, 11]])
X
Y = np.array(train.iloc[:, 1])
from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()
cabin_encoder = LabelEncoder()
X[:, 1] = sex_encoder.fit_transform(X[:, 1].astype(str))
X[:, 7] = embarked_encoder.fit_transform(X[:, 7].astype(str))
X[:, 6] = cabin_encoder.fit_transform(X[:, 6].astype(str))
from sklearn.preprocessing import StandardScaler
train_scale = StandardScaler()
X = train_scale.fit_transform(X)
X
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adadelta

model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y,epochs=1000)
test = pd.read_csv("../input/test.csv").fillna(0)
test
X_test = np.array(test.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10]])
X_test
X_test[0, :]
X_test[:, 1] = sex_encoder.transform(X_test[:, 1].astype(str))
X_test[:, 6] = cabin_encoder.fit_transform(X_test[:, 6].astype(str))
X_test[:, 7] = embarked_encoder.transform(X_test[:, 7].astype(str))
#test_scale = StandardScaler()
X_test = train_scale.transform(X_test)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
y_pred
test.shape
res = np.array(test.iloc[:, 0])
res
s = "PassengerId,Survived\n"
for i in range(len(res)):
    if(i==len(res) - 1):
        s = s + str(res[i]) + "," + str(int(y_pred[i]))
    else:
        s = s + str(res[i]) + "," + str(int(y_pred[i])) + "\n"
file = open("finale.csv", "w+")
file.write(s)
file.close()

print(s)

