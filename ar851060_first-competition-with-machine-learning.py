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
from sklearn.impute import SimpleImputer

median_imputer = SimpleImputer(strategy = 'median')

mostfreq_imputer = SimpleImputer(strategy = 'most_frequent')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

combine = [test_data, train_data]

train_data.head()
train_data.tail()
train_data.info()
train_data.describe()
train_data.isnull().sum()
train_data.describe(include=['O'])
cols = train_data.columns

dis_cols = cols.drop(['Survived', 'PassengerId','Name','Age','Ticket','Fare','Cabin'])

for x in dis_cols:

    ana = train_data[[x, 'Survived']].groupby(x).mean().sort_values(by='Survived',ascending=False)

    print(ana)

    print("_"*10)
import matplotlib.pyplot as plt

import seaborn as sns

g = sns.FacetGrid(train_data, col = "Survived", size = 10)

g.map(plt.hist, 'Age', bins = 50)

plt.xticks(np.arange(0,81, step = 5))

plt.show()

plt.figure(figsize = (14,6))

g = sns.FacetGrid(train_data, col = "Survived", size = 10)

g.map(plt.hist, 'Fare', bins = 20)

plt.xticks(np.arange(0,520, step = 50))

plt.show()
train_data.plot.scatter(x='Pclass', y='Fare', c='Survived', colormap='viridis')
train_data = train_data.drop(['Cabin','Ticket'], axis = 1)

test_data = test_data.drop(['Cabin', 'Ticket'], axis = 1)

combine = [train_data, test_data]

train_data.head()
train_data['Age'] = median_imputer.fit_transform(train_data[['Age']])

test_data['Age'] = median_imputer.transform(test_data[['Age']])

train_data['Fare'] = median_imputer.fit_transform(train_data[['Fare']])

test_data['Fare'] = median_imputer.transform(test_data[['Fare']])

train_data['Embarked'] = mostfreq_imputer.fit_transform(train_data[['Embarked']])

test_data['Embarked'] = mostfreq_imputer.transform(test_data[['Embarked']])
for df in combine:

    df['family'] = df['SibSp']+df['Parch']

train_data = train_data.drop(['SibSp','Parch'], axis = 1)

test_data = test_data.drop(['SibSp','Parch'],axis=1)

combine = [train_data, test_data]
train_data[['family', 'Survived']].groupby('family').mean().sort_values(by='Survived',ascending=False)
for df in combine:

    df['Alone'] = 0

    df.loc[df['family']==0, 'Alone']=1

train_data = train_data.drop(['family'],axis = 1)

test_data = test_data.drop(['family'],axis = 1)

combine = [train_data, test_data]

train_data[['Alone', 'Survived']].groupby('Alone').mean().sort_values(by='Survived',ascending=False)
for df in combine:

    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.',expand = False)



pd.crosstab(train_data['Title'], train_data['Sex'])
for df in combine:

    df['Title'] = df['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Mlle','Mme','Ms','Rev','Sir'],'Others')

train_data[['Title','Survived']].groupby(['Title']).mean()
train_data = train_data.drop(['Name'],axis=1)

test_data = test_data.drop(['Name'],axis = 1)

combine = [train_data,test_data]

for df in combine:

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

features = ['Embarked', 'Title']

OH_train_cols = pd.DataFrame(encoder.fit_transform(train_data[features]))

OH_test_cols = pd.DataFrame(encoder.transform(test_data[features]))

OH_train_cols.index = train_data.index

OH_test_cols.index = test_data.index

train_data = train_data.drop(features, axis = 1)

test_data = test_data.drop(features, axis = 1)

train_data = pd.concat([train_data, OH_train_cols], axis = 1)

test_data = pd.concat([test_data, OH_test_cols], axis = 1)
from sklearn.model_selection import GridSearchCV

X = train_data.drop(['PassengerId','Survived'], axis = 1)

y = train_data['Survived']

X_test = test_data.drop(['PassengerId'], axis =1).copy()
from sklearn.ensemble import RandomForestClassifier

grid_feature = {'max_features':[10,11,12,13]}

model = RandomForestClassifier(n_estimators = 1000,

                             random_state=1,

                             n_jobs=-1)

grid = GridSearchCV(model, param_grid = grid_feature)

grid.fit(X,y)

print(grid.best_params_)

print(grid.best_score_)
model = RandomForestClassifier(n_estimators = 1000,

                               max_features = 12,

                             random_state=1,

                             n_jobs=-1)

model.fit(X, y)

y_test = model.predict(X_test)

model.score(X, y)
submission = pd.DataFrame({"PassengerId":test_data['PassengerId'],"Survived":y_test})

submission.to_csv('titanic_submission.csv', index = False)