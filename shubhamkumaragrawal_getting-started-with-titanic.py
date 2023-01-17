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
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
train_data.describe()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
features=["Pclass","Sex", "Age","SibSp","Parch","Embarked"]

X=train_data[features]

X
y=train_data["Survived"]

y
test_data=test_data[features]

test_data
x_test=pd.get_dummies(test_data)

x_test
X=pd.get_dummies(X)

X
from sklearn.preprocessing import MinMaxScaler

scaled=MinMaxScaler()

X_scaled=scaled.fit_transform(X)

X=pd.DataFrame(X_scaled,columns=X.columns)

X
X.isnull().sum()
X['Age'].fillna(X['Age'].mean(), inplace=True)
X.isnull().sum()
x_test.isnull().sum()
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test.isnull().sum()
import keras

from keras.models import Sequential

from keras.layers import Dense
model= Sequential()

model.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform', input_dim=9))

model.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))

model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y,batch_size=10,epochs=1000,verbose=1)
y_pred=model.predict(x_test)

y_pred=(y_pred>0.5)

y_pred[:10]
train_pred=model.predict(X)

train_pred=(train_pred>0.5)

train_pred
from sklearn.metrics import accuracy_score

score=accuracy_score(train_pred,y)

score