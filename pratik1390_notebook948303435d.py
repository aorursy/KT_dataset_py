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

df_train = pd.read_csv("../input/train.csv")

df_test   = pd.read_csv("../input/test.csv")



# preview the data

df_test.info()

df_train["Embarked"] = df_train["Embarked"].fillna("S")

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train.columns

#df_train.groupby("Embarked")[['Fare']].std()

#sns.factorplot('Embarked','Fare', data=titanic_df,size=4,aspect=3)
sns.factorplot('Cabin','Survived', data=df_train,kind='bar',hue='Sex')
sns.factorplot('Sex','Survived', data=df_train,kind='bar')
def getChild(age):

    try:

        if age>16:

            return 1

        return 0

    except Exception:

        return 1

    

df_train["Adult"] = df_train.apply(lambda row:getChild(row["Age"]), axis=1)
#df_train[:3]

sns.factorplot('Sex','Survived', data=df_train,kind='bar',hue='Adult')
sns.factorplot('Sex','Survived', data=df_train,kind='bar')

print("xyz")
df_train["Age"].fillna(df_train["Age"].median(), inplace=True)

df_test["Age"].fillna(df_test["Age"].median(), inplace=True)

df_train["Age"].fillna(np.median(df_train["Age"]))

plt.subplot(2,1,1)

plt.hist(df_train[df_train["Survived"]==1]["Age"])

plt.subplot(2,1,2)

plt.hist(df_train[df_train["Survived"]==0]["Age"])



#sum(np.isnan(df_train.Age))

np.nanmedian(df_test["Fare"])
df_train["Fare"].fillna(np.nanmedian(df_train["Fare"]),inplace=True)

df_test["Fare"].fillna(np.nanmedian(df_test["Fare"]),inplace=True)

df_train['Fare'] = df_train['Fare'].astype(int)

df_test['Fare'] = df_test['Fare'].astype(int)

FareSurvived = df_train[df_train["Survived"]==1]["Fare"]

FareNotSurvived = df_train[df_train["Survived"]==0]["Fare"]
#plt.subplot(2,1,1)

#plt.hist(df_train[df_train["Survived"]==1]["Fare"])

#plt.subplot(2,1,2)

#plt.hist(df_train[df_train["Survived"]==0]["Fare"])

x=df_train[df_train["Survived"]==1]["Fare"]

y=df_train[df_train["Survived"]==0]["Fare"]

plt.hist(x, bins = np.arange(0,500,10),alpha=0.5, label='Survived')

plt.hist(y+200, bins = np.arange(0,500,10),alpha=0.5, label='Not Survived')

plt.legend(loc='upper right')

#sum(np.isnan(df_train.Age))

df_train.head()

df_train.drop(['Cabin','PassengerId','Name','Ticket'], axis=1, inplace=True)

df_test.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)

df_test.info()
embark_dummies_titanic  = pd.get_dummies(df_train['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(df_test['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



df_train = df_train.join(embark_dummies_titanic)

df_test    = df_test.join(embark_dummies_test)



df_train.drop(['Embarked'], axis=1,inplace=True)

df_test.drop(['Embarked'], axis=1,inplace=True)

titanic_df =df_train

test_df = df_test
df_test.info()
titanic_df =df_train

test_df = df_test

df_test.info()
# Family



# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

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



# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

titanic_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df    = test_df.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Person', data=titanic_df, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)

test_df.head()
# Pclass



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df    = test_df.join(pclass_dummies_test)
titanic_df.head()

# define training and testing sets



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
X_train.head()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



#logreg.score(X_train, Y_train)