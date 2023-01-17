# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pa

import numpy as np

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





traindata = pa.read_csv('../input/titanic/train.csv')

testdata = pa.read_csv('../input/titanic/test.csv')

# Any results you write to the current directory are saved as output.
testdata['Fare'].describe()
testdata['Age'].describe()
combineData = [traindata , testdata]
#cleaning data

for datainT in combineData:

    datainT['Title']=  datainT.Name.str.extract(' ([A-Za-z]+)\.' , expand=False)

    datainT['Title'] = datainT['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dr','Jonkheer','Dona','Lady','Major','Rev','Sir'], 'Rare')

    datainT['Title'] = datainT['Title'].replace(['Mlle', 'Ms'], 'Miss')

    datainT['Title'] = datainT['Title'].replace('Mme', 'Mrs')

    datainT['Title']=  datainT['Title'].map({"Mr":1, "Master":2,"Miss":3,"Mrs":4, "Rare":5}).astype(int)

    datainT['Sex']= datainT['Sex'].map({"male":1,"female":2}).astype(int)

    datainT['Embarked']= datainT['Embarked'].map({"S":1,"C":2})

    embmostfreqValue = datainT.Embarked.dropna().mode()[0]

    datainT['Embarked']= datainT['Embarked'].fillna(embmostfreqValue)

    datainT['Embarked'] = datainT['Embarked'].astype(int)

    ages_toguess=np.zeros((3,2))

    for i in range(1,4):

        for j in range(1,3):

            ageguessTemp=datainT[(datainT['Sex']==j)&(datainT['Pclass']==i)]['Age'].dropna().mean()

            datainT.loc[(datainT.Age.isnull()) & (datainT.Sex==j) & (datainT.Pclass==i),'Age'] = int(ageguessTemp)

    datainT['Age'] = pa.cut(datainT['Age'], bins=[0,12,20,40,120], labels=[1,2,3,4]).astype(int)

    datainT['isAlone'] = 0

    datainT.loc[(datainT.Parch >0) | (datainT.SibSp>0), 'isAlone']=1

    datainT["Fare"].fillna(datainT["Fare"].median(), inplace=True)

    datainT["Fare"] = datainT["Fare"].astype(int)

    datainT['Fare'] = pa.cut(datainT['Fare'], bins=[0,7.91,14.45,31,120], labels=[1,2,3,4]).astype(int)
#deleting files

for datainT in combineData:

    del datainT['Name']

    del datainT['Ticket']

    del datainT['Cabin']

    del datainT['SibSp']

    del datainT['Parch']



#     del datainT['Fare']

del traindata['PassengerId']
#checking in traindata %age of surviver

peopleSurvied = traindata[traindata['Survived']==1]['Survived'].count()

peopleNotSurvied = traindata[traindata['Survived']==0]['Survived'].count()

totalPeople = traindata['Survived'].count()



survivedPeoplePer = (peopleSurvied/ totalPeople) *100

survivedNotPeoplePer = (peopleNotSurvied/ totalPeople) *100



print(survivedPeoplePer , survivedNotPeoplePer)
#spliting data 

X_train = traindata.drop('Survived', axis=1)

X_test = testdata.drop("PassengerId", axis=1).copy()

Y_train = traindata['Survived']
#running randam forest 

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_Pred = random_forest.predict(X_test)

random_forest.score(X_train,Y_train)
submission = pa.DataFrame({

    "PassengerId" : testdata['PassengerId'],

    "Survived" : Y_Pred

})

submission.to_csv('titanic.csv', index=False)