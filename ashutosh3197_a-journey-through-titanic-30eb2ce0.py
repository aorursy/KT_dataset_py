import pandas as pd

import matplotlib as mpl

import numpy as np

import scipy as sp

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

from sklearn.ensemble import RandomForestClassifier
sns.barplot(x='Embarked',y='Survived',data=train)

embark_dummies=pd.get_dummies(train.Embarked)

embark_dummies.columns=['C','Q','S']

embark_dummies.drop('S',axis=1,inplace=True)

embark_dummiest=pd.get_dummies(test.Embarked)

embark_dummiest.columns=['C','Q','S']

embark_dummiest.drop('S',axis=1,inplace=True)

train.drop('Embarked',inplace=True,axis=1)

train=train.join(embark_dummies)

test=test.join(embark_dummiest)
fare_survived=train.Fare[train.Survived==1]

fare_notsurvived=train.Fare[train.Survived==0]
average_fare=pd.DataFrame([fare_survived.mean(),fare_notsurvived.mean()])

std_fare=pd.DataFrame([fare_survived.std(),fare_notsurvived.std()])

average_fare.index.names=std_fare.index.names=['Survived']
average_fare.plot(yerr=std_fare,kind='bar',legend=False)
new=train.Age[np.isnan(train['Age'])]

new_test=test.Age[np.isnan(test.Age)]

men=train.Age.mean()

stand=train.Age.std()

test_men=test.Age.mean()

test_stand=test.Age.std()

rand_int=np.random.randint(men-stand,men+stand,size=new.size)

rand_int2=np.random.randint(test_men-test_stand,test_men+test_stand,size=new_test.size)

train.Age[np.isnan(train.Age)]=rand_int

test.Age[np.isnan(test.Age)]=rand_int2

train.Age.hist(bins=70)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 12 else sex

    

train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)

test['Person']    = test[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

train.drop(['Sex'],axis=1,inplace=True)

test.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(train['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train = train.join(person_dummies_titanic)

test    = test.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Person', data=train, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



train.drop(['Person'],axis=1,inplace=True)

test.drop(['Person'],axis=1,inplace=True)
sns.factorplot('Pclass','Survived',data=train)

pclass_dummies=pd.get_dummies(train.Pclass)

pclass_dummies.columns=['class1','class2','class3']

pclass_dummies.drop(['class3'],axis=1,inplace=True)

pclass_dummiest=pd.get_dummies(test.Pclass)

pclass_dummiest.columns=['class1','class2','class3']

pclass_dummiest.drop(['class3'],axis=1,inplace=True)

train=train.join(pclass_dummies)

test=test.join(pclass_dummiest)

train.drop('Pclass',inplace=True,axis=1)
train['Family']=train['SibSp']+train['Parch']

train['Family'].loc[train['Family']>0]=1

train['Family'].loc[train['Family']==0]=0

test['Family']=test['SibSp']+test['Parch']

test['Family'].loc[test['Family']>0]=1

test['Family'].loc[test['Family']==0]=0

train.drop(['SibSp','Parch'],axis=1,inplace=True)

test.drop(['SibSp','Parch'],axis=1,inplace=True)

fig,(axis1,axis2)=plt.subplots(1,2)

sns.countplot(x='Family',data=train,ax=axis1)

family_perc=train[['Family','Survived']].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc,ax=axis2)
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

y    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

test.drop(['PassengerId','Pclass','Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)

y.head()
X_train=train.drop(['Survived'],axis=1,inplace=False)

Y_train=train.Survived

X_test=test.fillna(0)
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

test=test.join(y.PassengerId)

test.head()
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv("train.csv", index=False)

