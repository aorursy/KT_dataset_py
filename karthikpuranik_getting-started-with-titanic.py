# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

data=train_data.append(test_data,sort=False)
median=data['Age'].median()

train_data['Age'].fillna(median, inplace=True)

test_data['Age'].fillna(median, inplace=True)

test_data['Fare'].fillna(data['Fare'].median(),inplace=True)

train_data['Embarked'].mode()

data=train_data.append(test_data,sort=False)

data.head()
sex=pd.get_dummies(train_data["Sex"], drop_first=True)

pclass=pd.get_dummies(train_data["Pclass"],drop_first=True)

emb=pd.get_dummies(train_data["Embarked"],drop_first=True)

train_data=pd.concat([train_data,sex,pclass,emb],axis=1)

train_data.head()
sex=pd.get_dummies(test_data["Sex"], drop_first=True)

pclass=pd.get_dummies(test_data["Pclass"],drop_first=True)

emb=pd.get_dummies(test_data["Embarked"],drop_first=True)

test_data=pd.concat([test_data,sex,pclass,emb],axis=1)

test_data.head()
train_data["Adult"]=[1 if x>=18 else 0 for x in train_data["Age"]]

test_data["Adult"]=[1 if x>=18 else 0 for x in test_data["Age"]]
train_data.drop(["Pclass","Name","Sex","Ticket","Cabin","Embarked","Age"],axis=1, inplace=True)

test_data.drop(["Pclass","Name","Sex","Ticket","Cabin","Embarked","Age"],axis=1, inplace=True)
data=train_data.append(test_data,sort=False)

data.head()
corr_matrix=data.corr().round(2)

sns.heatmap(data=corr_matrix, annot=True)
X_train=pd.DataFrame(train_data.drop(["Survived","PassengerId","Fare",2,"Q","S"],axis=1))

y_train=pd.DataFrame(train_data["Survived"])

X_test=pd.DataFrame(test_data)

X_test.drop(["PassengerId","Fare",2,"Q","S"],axis=1,inplace=True)

X_test.head()
y_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



#initialising ANN

classifier=Sequential()



#adding input and first layer

classifier.add(Dense(output_dim= 6, init='uniform', activation='relu',

                    input_dim=5))

classifier.add(Dropout(0.1))



#adding another layer

classifier.add(Dense(output_dim= 6, init='uniform', activation='relu'))

classifier.add(Dropout(0.1))

classifier.add(Dense(output_dim= 6, init='uniform', activation='relu'))

classifier.add(Dropout(0.1))

#adding output layer

classifier.add(Dense(output_dim= 1, init='uniform', activation='sigmoid'))



#compiling ANN

classifier.compile(optimizer='adam', loss='binary_crossentropy', 

                   metrics=['accuracy'])



classifier.fit(X_train, y_train, batch_size=50

               , epochs=500)
predictions = classifier.predict(X_test)

new_y_pred = []

for var in predictions:

    if var>=0.7:

        new_y_pred.append(1)

    else:

        new_y_pred.append(0)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': new_y_pred})

output.to_csv('my_submission.csv', index=False)