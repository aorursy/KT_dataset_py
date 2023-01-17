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
dataset = pd.read_csv('../input/train.csv')

testset = pd.read_csv ('../input/test.csv')





dataset.head()

dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

dataset['Embarked']= dataset['Embarked'].fillna(dataset['Embarked'].value_counts().index[0])



dataset.isnull().sum()
testset['Age'] = testset['Age'].fillna(testset['Age'].median())

testset['Fare'] = testset['Fare'].fillna(testset['Fare'].median())

testset['Embarked']= testset['Embarked'].fillna(testset['Embarked'].value_counts().index[0])

testset.isnull().sum()
features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

x = dataset[features]

y = dataset['Survived']

X = testset[features]



from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

x['Sex'] = LE.fit_transform(x['Sex'])

x['Embarked'] = LE.fit_transform(x['Embarked'])



X['Sex'] = LE.fit_transform(X['Sex'])

X['Embarked'] = LE.fit_transform(X['Embarked'])







from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state =0)
from keras.models import Sequential

from keras.layers import Dense







model = Sequential()

model.add(Dense(14, input_dim=7, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(10, activation='sigmoid'))



model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x_train, y_train , epochs=30, batch_size=3)

x_test['Age'] = x_test['Age'].fillna(x_test['Age'].median())

x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].median())

scores = model.evaluate(x_test, y_test)



scores
prediction = model.predict(X)

prediction[1][0]

pdf=[]

for i in range (len(prediction)):

    if (prediction[i][0]>0.6):

        pdf.append(1)

    else :

        pdf.append(0)

    

pdf

#testset['PassengerId']

output = pd.DataFrame({'PassengerId': testset['PassengerId'],'Survived':pdf})

output.to_csv('submission.csv', index=False)

output