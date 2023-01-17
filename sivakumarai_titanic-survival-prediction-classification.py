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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression



titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")

print("shape of Titanic_train:",titanic_train.shape)

print("shape of Titanic_test:",titanic_test.shape)

display(titanic_test)

titanic_train['From'] = 'TRAIN'

titanic_test['From'] = 'TEST'

titanic = pd.concat([titanic_train,titanic_test],axis=0,sort=False)

titanic.head()

    

    

    



ax = sns.countplot(titanic['Pclass'])
print("Mean:",titanic['Age'].mean())

print("Median:",titanic['Age'].median())

print((titanic['Age'].mean() + titanic['Age'].std()), (titanic['Age'].mean() + (2 * titanic['Age'].std())), (titanic['Age'].mean() + (3 * titanic['Age'].std())))

titanic = titanic[titanic['Age'] <= 73]

titanic['Age'].fillna(titanic['Age'].median(),inplace=True)

sns.distplot(titanic['Age'], bins=8)

print("Mean:",titanic['Age'].mean())

print("Median:",titanic['Age'].median())



display(titanic.corr()['Age'])
sns.distplot(titanic['Fare'],bins=20)
print(titanic['Fare'].mean())

print(titanic['Fare'].median())

print((titanic['Fare'].mean() + titanic['Fare'].std()),(titanic['Fare'].mean() + (2 * titanic['Fare'].std())),(titanic['Fare'].mean() + (3 * titanic['Fare'].std())))

print("Minimum:", titanic['Fare'].min())

print("Maximum:", titanic['Fare'].max())

titanic = titanic[titanic['Fare'] <= 202]

print("Minimum:", titanic['Fare'].min())

print("Maximum:", titanic['Fare'].max())

print(titanic['Fare'].mean())

print(titanic['Fare'].median())

sns.boxplot(titanic['Fare'])

print(titanic['Fare'].isnull().sum())
sns.distplot(titanic['Fare'], bins=8)
print(titanic.shape)
sns.countplot('Sex',hue='Survived',data=titanic)

titanic['Name'] = titanic['Name'].apply(lambda x: "Mrs" if "Mrs" in x else ("Mr" if "Mr" in x else ("Miss")))

display(titanic.head(500))

display(titanic['Name'].isnull().sum())
sns.countplot("Name",hue="Survived", data=titanic)
titanic['Name'].replace({"Miss":3,"Mrs":2,"Mr":1},inplace=True)

titanic.head()

print("PClass counts:")

display(titanic.groupby(['Pclass','Survived'])['Survived'].count()/len(titanic)*100)

print("Sex counts:")

display(titanic.groupby(['Sex','Survived'])['Survived'].count()/len(titanic)*100)
print("Embarked counts:")

display(titanic.groupby(['Embarked','Survived'])['Survived'].count()/len(titanic)*100)
titanic = titanic.drop(['Ticket'],axis=1)

titanic.head()
titanic['SibSp_Parch'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic.drop(columns=['SibSp','Parch'],axis=1,inplace=True)

titanic.head()
display(titanic['Age'].isnull().sum())

display(titanic.groupby('Sex')['Age'].mean())

#display(titanic.groupby('Sex')['Age'].mean()[1])

titanic['Age'].fillna(titanic.groupby('Sex')['Age'].mean()[0],inplace=True)

display(titanic['Age'].isnull().sum())

display(titanic['Embarked'].isnull().sum())

display(titanic.groupby('Embarked')['Embarked'].count())

titanic['Embarked'].fillna('S',inplace=True)

titanic.head()
sns.distplot(titanic['Age'])
from sklearn.preprocessing import StandardScaler

#titanic['Age'] = titanic['Age'].apply(lambda x: int(x))

#titanic['Age'] = titanic['Age'].apply(lambda x: 3 if x <= 20 else (2 if x <= 60 else (1)))

#ax = sns.distplot(titanic['Age'])

scaler = StandardScaler()

Age_scale = scaler.fit_transform(titanic[['Age']])



print(Age_scale)

ax = sns.distplot(Age_scale)
display(titanic['Fare'].isnull().sum())

titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].mean())

display(titanic['Fare'].isnull().sum())

scaler = StandardScaler()

Fare_scale = scaler.fit_transform(titanic[['Fare']])

print(Fare_scale)
ax = sns.distplot(titanic['Fare'])
ax = sns.distplot(Fare_scale)
titanic.head()

titanic['Sex'].replace({"male":1,"female":2}, inplace=True)

titanic['Embarked'].replace({"C":2,"Q":1,"S":3}, inplace=True)

titanic['Pclass'].replace({1:3,2:2,3:1},inplace=True)

#titanic.drop('Fare',axis=1,inplace=True)

titanic.head()

#titanic['Cabin'] = titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])

display(titanic['Cabin'].isnull().sum())

#titanic.head()

titanic.drop('Cabin',axis=1,inplace=True)

titanic.head()
#titanic = pd.get_dummies(titanic)

#titanic.head()
print(titanic.isnull().sum())
display(titanic_train.shape)

titanic_train = titanic[titanic['From'] == 'TRAIN']

titanic_test = titanic[titanic['From'] == 'TEST']

display(titanic_train.shape)

display(titanic_test.shape)

titanic_test.drop(columns=['Survived','From'],inplace=True)

display(titanic_test.shape)

print(titanic_train.isnull().sum())

print(titanic_test.isnull().sum())

X = titanic_train.drop(['Survived','PassengerId','From'],axis=1)

print(X.isnull().sum())

Y = titanic_train['Survived']

print(Y.isnull().sum())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
#scaler = StandardScaler()

#X_train = scaler.fit_transform(X_train)

##display(X_train)

from sklearn.feature_selection import SelectKBest

select_feature = SelectKBest(chi2,k=4).fit(X_train,Y_train)

selected_features_df = pd.DataFrame({"Feature": list(X_train.columns),"Scores": select_feature.scores_})

selected_features_df.sort_values(by='Scores',ascending=False,inplace=True)

display(selected_features_df)

from sklearn.metrics import accuracy_score

X_train.drop(["SibSp_Parch",'Age'],axis=1,inplace=True)

X_test.drop(["SibSp_Parch",'Age'],axis=1,inplace=True)

titanic_test1 = titanic_test.copy()

titanic_test1.drop(['PassengerId','SibSp_Parch','Age'],axis=1,inplace=True)

log_model = LogisticRegression()

log_model.fit(X_train,Y_train)

Y_train_predict = log_model.predict(X_train)

Y_test_predict = log_model.predict(X_test)

Final_Y_predict = log_model.predict(titanic_test1)

print("Accuracy score train:", accuracy_score(Y_train,Y_train_predict))

print("Accuracy score test:", accuracy_score(Y_test,Y_test_predict))
from sklearn.ensemble import ExtraTreesClassifier



extra = ExtraTreesClassifier()

extra.fit(X_train,Y_train)

display(X.head())

print("Feature Importances:", extra.feature_importances_)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

depth_range = range(1,10)

leaf_range = range(1,15)

param_grid = {"max_depth":depth_range, "min_samples_leaf":leaf_range}

ds_model = DecisionTreeClassifier()

grid = GridSearchCV(ds_model,param_grid,cv=10,scoring="accuracy")

grid.fit(X_train,Y_train)



ds_model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=3, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

ds_model.fit(X_train,Y_train)

Y_train_predict = ds_model.predict(X_train)

Y_test_predict = ds_model.predict(X_test)

Final_Y_predict = log_model.predict(titanic_test1)

print("Accuracy score train:", accuracy_score(Y_train,Y_train_predict))

print("Accuracy score test:", accuracy_score(Y_test,Y_test_predict))

#print(ds_model.feature_importances_)

print("best score:", grid.best_score_)

print("best params:", grid.best_params_)

print("best estimator:", grid.best_estimator_)

Final_df = pd.DataFrame({"PassengerId":titanic_test['PassengerId'],"Survived":Final_Y_predict})

Final_df['Survived'] = Final_df['Survived'].apply(lambda x: int(x))

pd.set_option('display.max_rows',500)

display(Final_df.head(417))

Final_df.to_csv("Titanic_Submission.csv",index=False)

                         