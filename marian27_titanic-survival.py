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
#datafile='../input/train.csv'

#data=pd.read_csv(datafile)

#data.head()
def preprocessing(data):

    data['Gender']=data['Sex'].apply(lambda a: 1 if a=='male' else 0)

    data=data.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex'],axis=1)

    data=data.dropna()

    return data

data=pd.read_csv("../input/train.csv")

train=preprocessing(data)





test=pd.read_csv("../input/test.csv")

PassengerID=test['PassengerId']

test['Gender']=test['Sex'].apply(lambda a: 1 if a=='male' else 0)

test=test.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex'],axis=1)



test.head()


train.head()
test.isnull().sum()
test['Age'].head().plot.density()
data[data['Age']>50].plot.density(x='Age',y='Survived')
test['Age']=test['Age'].fillna(test['Age'].mean())

test['Fare']=test['Fare'].fillna(test['Fare'].mean())

test.head()
data['Age']=data['Age'].fillna(data['Age'].mean())

data['Fare']=data['Fare'].fillna(data['Fare'].mean())

data.head()
#test['Gender']=test['Sex'].apply(lambda a: 1 if a=='male' else 0)



#test=test.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex'],axis=1)



#test=test.dropna()



#test.head()
from sklearn.ensemble import RandomForestClassifier

x_train=train.drop(['Survived'],axis=1).values

y_train=train['Survived'].values

x_test=test.values



model=RandomForestClassifier(n_estimators=1000,max_leaf_nodes=15)

model.fit(x_train,y_train)

predictions=model.predict(x_test)

predictions
x_test

#model=RandomForestClassifier(n_estimators=1000,max_leaf_nodes=15)

#model.fit(x_train,y_train)

#predictions=model.predict(x_test)

#predictions
len(PassengerID)
len(predictions)
resultFinal=pd.DataFrame({'PassengerID':PassengerID,'Survived':predictions})

resultFinal.head()
resultFinal.to_csv('Survival_Submission.csv',index=False)
print(".../output/Working")
print(os.listdir("../working"))
a=pd.read_csv("../input/gender_submission.csv")

a.head(1)
resultFinal.to_csv('survival2_submission.csv',index=False)
print(os.listdir("../working"))
b=pd.read_csv("survival2_submission.csv")

b.head(5)