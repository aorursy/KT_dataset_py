# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trainData = pd.read_csv('../input/titanic/train.csv')

testData = pd.read_csv('../input/titanic/test.csv')
#removing unwanted cols from train

del trainData['Name']

del trainData['Ticket']

trainData['totalMembers'] = trainData['SibSp'] + trainData['Parch']

del trainData['SibSp']

del trainData['Parch']

del trainData['Cabin']

del trainData['Embarked']

trainData.head(20)
del testData['Name']

del testData['Ticket']

testData['totalMembers'] = testData['SibSp'] + testData['Parch']

del testData['SibSp']

del testData['Parch']

del testData['Cabin']

del testData['Embarked']

testData.head(20)
trainData.isnull().sum()
testData.isnull().sum()
#cleaning Pclass for train Data

Pclass_1_avg_age = trainData[trainData['Pclass']==1]['Age'].median()

Pclass_2_avg_age = trainData[trainData['Pclass']==2]['Age'].median()

Pclass_3_avg_age = trainData[trainData['Pclass']==3]['Age'].median()



def fill_age(age):

    if str(age[5]).lower()=='nan':

        if age[2]==1:

            return Pclass_1_avg_age

        elif age[2]==2:

            return Pclass_2_avg_age

        else:

            return Pclass_3_avg_age

    else:

        return age[5]



trainData['Age']=trainData.apply(fill_age,axis=1)
#cleaning Pclass for test Data

Pclass_1_avg_age = testData[testData['Pclass']==1]['Age'].median()

Pclass_2_avg_age = testData[testData['Pclass']==2]['Age'].median()

Pclass_3_avg_age = testData[testData['Pclass']==3]['Age'].median()



def fill_age_test(age):

    if str(age[5]).lower()=='nan':

        if age[2]==1:

            return Pclass_1_avg_age

        elif age[2]==2:

            return Pclass_2_avg_age

        else:

            return Pclass_3_avg_age

    else:

        return age[5]



testData['Age']=testData.apply(fill_age_test,axis=1)

testData['Fare'].fillna(testData['Fare'].median(),inplace=True)
trainData.isnull().sum()
testData.isnull().sum()
sns.countplot(x = "totalMembers", hue = "Survived", data = trainData)
sns.countplot(x = "Pclass", hue = "Survived", data = trainData)
sns.countplot(x = "Sex", hue = "Survived", data = trainData)
sns.countplot(x = "Age", hue = "Survived", data = trainData)
sns.countplot(x = "Fare", hue = "Survived", data = trainData)
#encoding sex to 0-1

le = LabelEncoder()

trainData['Sex']=le.fit_transform(trainData.Sex.values)

testData['Sex']=le.fit_transform(testData.Sex.values)
#Training Data

features = ["Pclass","Age","Sex","Fare","totalMembers"]

yTrain = trainData['Survived']

xTrain = trainData[features]

model = RandomForestClassifier()

model.fit(xTrain,yTrain)

model.score(xTrain,yTrain)
xTrain.head()
xTest = testData[features]

xTest.head()

pred = model.predict(xTest)
my_submission = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': pred})
my_submission.to_csv('submission.csv', index=False)