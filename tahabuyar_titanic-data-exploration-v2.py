import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
train_data.shape
test_data.shape
train_data.info()

test_data.info()
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
sns.countplot(train_data["Survived"], hue=train_data["Sex"])
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.countplot(train_data["Survived"], hue=train_data["Embarked"])
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.countplot(train_data["Survived"], hue=train_data["Pclass"])
sns.catplot(x="Survived",y="Fare",data=train_data, kind="boxen")
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Fare', bins=20)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=15)
sns.countplot(train_data["Survived"], hue=train_data["Parch"])
sns.countplot(train_data["Survived"], hue=train_data["SibSp"])
train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.plotting.scatter_matrix(train_data, figsize=(12,12));
train_data.groupby(train_data["Survived"]).hist(figsize=(6,8))
cor=train_data.corr()

cor_target=abs(cor["Survived"]).sort_values(ascending=False)

cor_target
sns.heatmap(cor, annot=True, fmt=".2f")
X=train_data[["Pclass", "Age", "SibSp", "Parch", "Fare"]]

y=train_data["Survived"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier()

rfc.fit(X_train, y_train)

rfc.score(X_test,y_test)
train_data.info()

test_data.info()
train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)

test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)

test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True)
X=train_data[["Pclass", "Age", "SibSp", "Parch", "Fare"]]

y=train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc=RandomForestClassifier(random_state=35)

rfc.fit(X_train, y_train)



print("test accuracy: ",rfc.score(X_test,y_test))
train_data["Sex_encoded"]=pd.factorize(train_data["Sex"])[0]

train_data["Embarked_encoded"]=pd.factorize(train_data["Embarked"])[0]



test_data["Sex_encoded"]=pd.factorize(test_data["Sex"])[0]

test_data["Embarked_encoded"]=pd.factorize(test_data["Embarked"])[0]
train_data.info()

test_data.info()
train_data["Embarked"].unique()
train_data["Embarked_encoded"].unique()
X=train_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_encoded", "Embarked_encoded"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc=RandomForestClassifier(random_state=35)

rfc.fit(X_train, y_train)



print("test accuracy: ",rfc.score(X_test,y_test))
train_data["Cabin"].unique()
train_data["Cabin"].fillna("N", inplace=True)

train_data['Cabin_code'] = train_data["Cabin"].str.slice(0,1)

train_data['Cabin_code'].unique()
test_data["Cabin"].fillna("N", inplace=True)

test_data['Cabin_code'] = test_data["Cabin"].str.slice(0,1)
train_data["Cabin_code_encoded"]=pd.factorize(train_data["Cabin_code"])[0]



test_data["Cabin_code_encoded"]=pd.factorize(test_data["Cabin_code"])[0]
train_data.info()

test_data.info()
X=train_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_encoded", "Embarked_encoded", "Cabin_code_encoded"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc=RandomForestClassifier(random_state=35)

rfc.fit(X_train, y_train)



print("test accuracy: ",rfc.score(X_test,y_test))
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

X_train_sc=scaler.fit_transform(X_train)

X_test_sc=scaler.transform(X_test)
rfc=RandomForestClassifier(random_state=35)

rfc.fit(X_train_sc, y_train)

print("train accuracy: ",rfc.score(X_train_sc, y_train))

print("test accuracy: ",rfc.score(X_test_sc,y_test))
rfc=RandomForestClassifier(random_state=35)

for x in X_train.columns:

    

    rfc.fit(X_train[[x]], y_train)

    print(x,"train accuracy: ",rfc.score(X_train[[x]], y_train)*100)

    print(x,"test accuracy: ",rfc.score(X_test[[x]],y_test)*100)
train_data["Age"].groupby(train_data["Survived"]).plot(kind="hist", bins=20, legend=True, alpha=0.5)
train_data.loc[train_data['Age'] <= 7.5, 'Age_encoded'] = 0

train_data.loc[(train_data['Age'] > 7.5) & (train_data['Age'] <= 15), 'Age_encoded'] = 1

train_data.loc[(train_data['Age'] > 15) & (train_data['Age'] <= 25), 'Age_encoded'] = 2

train_data.loc[(train_data['Age'] > 25) & (train_data['Age'] <= 30), 'Age_encoded'] = 3

train_data.loc[(train_data['Age'] > 30) & (train_data['Age'] <= 35), 'Age_encoded'] = 4

train_data.loc[(train_data['Age'] > 35) & (train_data['Age'] <= 50), 'Age_encoded'] = 5

train_data.loc[train_data['Age'] > 50, 'Age_encoded'] = 6

train_data["Age_encoded"].unique()
test_data.loc[test_data['Age'] <= 7.5, 'Age_encoded'] = 0

test_data.loc[(test_data['Age'] > 7.5) & (test_data['Age'] <= 15), 'Age_encoded'] = 1

test_data.loc[(test_data['Age'] > 15) & (test_data['Age'] <= 25), 'Age_encoded'] = 2

test_data.loc[(test_data['Age'] > 25) & (test_data['Age'] <= 30), 'Age_encoded'] = 3

test_data.loc[(test_data['Age'] > 30) & (test_data['Age'] <= 35), 'Age_encoded'] = 4

test_data.loc[(test_data['Age'] > 35) & (test_data['Age'] <= 50), 'Age_encoded'] = 5

test_data.loc[test_data['Age'] > 50, 'Age_encoded'] = 6

test_data["Age_encoded"].unique()
train_data[['Age_encoded', 'Survived']].groupby(['Age_encoded'], as_index=False).mean()
sns.countplot(train_data["Survived"], hue=train_data["Age_encoded"])
train_data["Fare"].groupby(train_data["Survived"]).plot(kind="hist", bins=20, legend=True, alpha=0.5)
train_data.loc[train_data['Fare'] <= 12.5, 'Fare_encoded'] = 0

train_data.loc[(train_data['Fare'] > 12.5) & (train_data['Fare'] <= 25), 'Fare_encoded'] = 1

train_data.loc[(train_data['Fare'] > 25) & (train_data['Fare'] <= 50), 'Fare_encoded'] = 2

train_data.loc[(train_data['Fare'] > 50) & (train_data['Fare'] <= 75), 'Fare_encoded'] = 3

train_data.loc[(train_data['Fare'] > 75) & (train_data['Fare'] <= 100), 'Fare_encoded'] = 4

train_data.loc[(train_data['Fare'] > 100) & (train_data['Fare'] <= 150), 'Fare_encoded'] = 5

train_data.loc[train_data['Fare'] > 150, 'Fare_encoded'] = 6

train_data["Fare_encoded"].unique()
test_data.loc[test_data['Fare'] <= 12.5, 'Fare_encoded'] = 0

test_data.loc[(test_data['Fare'] > 12.5) & (test_data['Fare'] <= 25), 'Fare_encoded'] = 1

test_data.loc[(test_data['Fare'] > 25) & (test_data['Fare'] <= 50), 'Fare_encoded'] = 2

test_data.loc[(test_data['Fare'] > 50) & (test_data['Fare'] <= 75), 'Fare_encoded'] = 3

test_data.loc[(test_data['Fare'] > 75) & (test_data['Fare'] <= 100), 'Fare_encoded'] = 4

test_data.loc[(test_data['Fare'] > 100) & (test_data['Fare'] <= 150), 'Fare_encoded'] = 5

test_data.loc[test_data['Fare'] > 150, 'Fare_encoded'] = 6

test_data["Fare_encoded"].unique()
train_data[['Fare_encoded', 'Survived']].groupby(['Fare_encoded'], as_index=False).mean()
sns.countplot(train_data["Survived"], hue=train_data["Fare_encoded"])
X=train_data[["Pclass", "Age_encoded", "SibSp", "Parch", "Fare_encoded", "Sex_encoded", "Embarked_encoded", "Cabin_code_encoded"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc=RandomForestClassifier(random_state=35)

rfc.fit(X_train, y_train)

print("train accuracy: ",rfc.score(X_train, y_train))

print("test accuracy: ",rfc.score(X_test,y_test))
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time
rfc_parameters = { 

    'n_estimators': [100,200,500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [6,8,10],

    'criterion' :['gini', 'entropy'],

    'min_samples_split': [2, 4, 6]

}
start_time = time.time()



rand_search= RandomizedSearchCV(rfc, rfc_parameters, cv=5)

rand_search.fit(X_train, y_train)

print(rand_search.best_params_)

print("best accuracy :",rand_search.best_score_)



end_time = time.time()

print("Total execution time: {} seconds".format(end_time - start_time))
start_time = time.time()



grid_search= GridSearchCV(rfc, rfc_parameters, cv=5)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

print("best accuracy :",grid_search.best_score_)



end_time = time.time()

print("Total execution time: {} seconds".format(end_time - start_time))
test_data.info()
X_test_final=test_data[["Pclass", "Age_encoded", "SibSp", "Parch", "Fare_encoded", "Sex_encoded", "Embarked_encoded", "Cabin_code_encoded"]]
prediction=rand_search.best_estimator_.predict(X_test_final)

output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Survived': prediction})

output.to_csv('submission.csv', index=False)