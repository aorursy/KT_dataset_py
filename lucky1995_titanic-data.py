# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.head()
titanic_df.info()

print("______________-------------------******************")

test_df.info()
titanic_df= titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df = test_df.drop(['PassengerId','Name','Ticket'],axis=1)
titanic_df['Embarked']= titanic_df['Embarked'].fillna('S')
sns.factorplot(data=titanic_df, x='Embarked' , y='Survived')
sns.countplot(x='Embarked', data=titanic_df)
sns.countplot(x='Survived',hue='Embarked',data=titanic_df, order=[1,0])
emMean= titanic_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked',y='Survived',data=emMean,order=['S','C','Q'])
#embarked

embarkDummy_titanic= pd.get_dummies(titanic_df['Embarked'])

embarkDummy_titanic.drop(['S'],axis=1,inplace=True)



embarkDummy_test= pd.get_dummies(test_df['Embarked'])

embarkDummy_test.drop(['S'], axis=1, inplace= True)



titanic_df = titanic_df.join(embarkDummy_titanic)

test_df= test_df.join(embarkDummy_test)



titanic_df.drop(['Embarked'],axis=1,inplace=True)

test_df.drop(['Embarked'],axis=1,inplace=True)
#fare

test_df['Fare'].fillna(test_df['Fare'].median(),inplace= True)



fare_survived= titanic_df['Fare'][titanic_df['Survived']==1]

fare_not_survived= titanic_df['Fare'][titanic_df['Survived']==0]



avg= DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std= DataFrame([fare_not_survived.std(), fare_survived.std()])



avg.index.names = std.index.names = ["Survived"]

avg.plot(yerr=std,kind='bar',legend=False)
#Age

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



titanic_age_mean= titanic_df['Age'].mean()

titanic_age_std= titanic_df['Age'].std()

titanic_nan_count= titanic_df['Age'].isnull().sum()

 

test_age_mean= test_df['Age'].mean()

test_age_std= test_df['Age'].std()

test_nan_count= test_df['Age'].isnull().sum()



titanic_df['Age'].dropna().astype(int).hist(bins=100, ax= axis1)



rand1= np.random.randint(titanic_age_mean - titanic_age_std, titanic_age_mean +titanic_age_std ,size=titanic_nan_count)

rand2= np.random.randint(test_age_mean - test_age_std,  test_age_mean + test_age_std ,size=test_nan_count)



titanic_df['Age'][np.isnan(titanic_df['Age'])]=rand1

test_df['Age'][np.isnan(test_df['Age'])]= rand2





titanic_df['Age']= titanic_df['Age'].astype(int)

test_df['Age']= test_df['Age'].astype(int)



titanic_df['Age'].hist(bins=100,ax= axis2)

#test_df.hist(bins=100)
# Cabin

titanic_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)



#family



titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1

titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# drop Parch & SibSp

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)



# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

sns.countplot(x='Family',data= titanic_df, order= [1,0] ,ax=axis1)



family_group= titanic_df[['Family','Survived']].groupby('Family', as_index= False).mean()

sns.barplot(x="Family", y= 'Survived', data= family_group, order=[1,0],ax=axis2)

#sex

fig, (axis1, axis2)= plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Sex',data=titanic_df,ax=axis1)

sns.countplot(x='Sex',data=test_df,ax=axis2)



def get_person(person):

    sex, age= person

    return "child" if age< 14 else sex



titanic_df['person']= titanic_df[['Sex','Age']].apply(get_person , axis=1)

test_df['person']= titanic_df[['Sex','Age']].apply(get_person, axis=1 )



titanic_df.drop('Sex', axis=1, inplace= True)

test_df.drop('Sex', axis=1 , inplace= True)
#continue sex



person_dummies_titanic= pd.get_dummies(titanic_df['person'])

person_dummies_titanic_columns= ['child', 'female','male']

person_dummies_titanic.drop(['male'], axis=1, inplace= True)



person_dummies_test= pd.get_dummies(test_df['person'])

person_dummies_test_columns= ['child', 'female','male']

person_dummies_test.drop(['male'], axis=1, inplace= True)



titanic_df.join(person_dummies_titanic)

test_df.join(person_dummies_test)



fig, (axis1, axis2)= plt.subplots(1,2, figsize= (10,5))



sns.countplot(x='person', data= titanic_df, ax=axis1)



person_group= titanic_df[['person','Survived']].groupby('person', as_index=False).mean()

sns.barplot(x='person',y='Survived',data=person_group,order=['child','female','male'],ax= axis2)



titanic_df.drop(['person'],axis=1,inplace=True)

test_df.drop(['person'],axis=1,inplace=True)
#pcclass

pc_group= titanic_df[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()

sns.barplot(x='Pclass',y='Survived',data=pc_group)





Pclass_dummies_titanic= pd.get_dummies(titanic_df['Pclass'])

Pclass_dummies_titanic_columns= ['class1','class2','class3']

Pclass_dummies_titanic.drop(['class3'],axis=1,inplace=True)



Pclass_dummies_test= pd.get_dummies(test_df['Pclass'])

Pclass_dummies_test_columns= ['class1','class2','class3']

Pclass_dummies_test.drop(['class3'],axis=1,inplace=True)



titanic_df= titanic_df.join(Pclass_dummies_titanic)

test_df= test_df.join(Pclass_dummies_test)



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)
X_train= titanic_df.drop('Survived',axis=1)

Y_train= titanic_df['Survived']



X_test= test_df
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
