import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
#removing unwanted cols from train
del trainData['Name']
del trainData['Ticket']
trainData['totalMembers'] = trainData['SibSp'] + trainData['Parch']
del trainData['SibSp']
del trainData['Parch']
del trainData['Cabin']
del trainData['Embarked']
trainData.head(20)
#removing unwanted cols from test
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