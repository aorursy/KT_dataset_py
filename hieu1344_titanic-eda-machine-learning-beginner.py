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
%matplotlib inline

#Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



#machine learning

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
#load the data

train_df = pd.read_csv('../input/train.csv', index_col=0)

test_df = pd.read_csv('../input/test.csv', index_col=0)

print(train_df.shape)

print(test_df.shape)
train_df.columns
train_df.info()
train_df.describe()
train_df.head()
train_df.loc[:, ['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.loc[:, ['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
#Create new feature

def age_feature(age):

    if age < 30:

        return 1

    elif age < 55:

        return 2

    elif age >=55:

        return 3

    

train_df['Age feature'] = train_df['Age'].apply(age_feature)

pd.crosstab(train_df['Survived'], train_df['Age feature'])
train_df[train_df['Survived'] == 1]['Age'].hist(color="green", 

                                         label='Survived', alpha=.5

                                       )

train_df[train_df['Survived'] == 0]['Age'].hist(color="red", 

                                         label='Died', alpha=.5,

                                       )

plt.title('Age for survived and died')

plt.xlabel('Years')

plt.ylabel('Frequency')

plt.legend();
sns.countplot(x=train_df['Age feature'], hue=train_df['Survived'])
train_df.drop(['Name', 'Ticket', 'Cabin', 'Age feature'], axis=1, inplace=True)

test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
X_train = train_df.drop('Survived', axis=1)

y_train = train_df['Survived']

X_test = test_df

X_train.shape, y_train.shape, X_test.shape
class ColumnSelectTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        if not isinstance(X, pd.DataFrame):

            X = pd.DataFrame(X)

        return X[self.columns].values

sex_encoder = Pipeline([('cst', ColumnSelectTransformer(['Sex'])),

                        ('encoder', OneHotEncoder(sparse=False))])

embarked_encoder = Pipeline([('cst', ColumnSelectTransformer(['Embarked'])),

                        ('encoder', OneHotEncoder(sparse=False))])

age_imputer = Pipeline([('cst', ColumnSelectTransformer(['Age'])),

                        ('imputer', Imputer(strategy='median'))])
X_train['Age'] = age_imputer.fit_transform(X_train)

X_test['Age'] = age_imputer.fit_transform(X_test)
X_train['Embarked'] = X_train['Embarked'].fillna(method='ffill')
X_train['Sex'] = sex_encoder.fit_transform(X_train)

X_train['Embarked'] = embarked_encoder.fit_transform(X_train)

X_test['Sex'] = sex_encoder.fit_transform(X_test)

X_test['Embarked'] = embarked_encoder.fit_transform(X_test)
param_grid = {'penalty':['l1', 'l2'],

              'tol' : np.linspace(1e-9, 1e-4, 10),

              }

lg_grid = GridSearchCV(LogisticRegression(), param_grid, n_jobs=2, cv=5, verbose=1)

lg_grid.fit(X_train, y_train)

log_reg = lg_grid.best_estimator_

print('LogisticRegression score: {}'.format(log_reg.score(X_train, y_train)))
param_grid = {'C':np.logspace(-3, 2, 25),}

svc_grid = GridSearchCV(SVC(kernel='sigmoid', probability=True), param_grid, n_jobs=2, cv=5)

svc_grid.fit(X_train, y_train)

svc = svc_grid.best_estimator_

print(svc)

print('SupportVectorClassifier score: {}'.format(log_reg.score(X_train, y_train)))
param_grid = {'min_samples_split': range(2, 15), 'min_samples_leaf': range(1, 15)}

tree_grid =  GridSearchCV(DecisionTreeClassifier(), param_grid, n_jobs=2, cv=5)

tree_grid.fit(X_train, y_train)

tree = tree_grid.best_estimator_

print(tree)

print('DecisionTree score: {}'.format(log_reg.score(X_train, y_train)))
forest = RandomForestClassifier(n_estimators=100, n_jobs=2)

forest.fit(X_train, y_train)

print('RandomForest score: {}'.format(log_reg.score(X_train, y_train)))
param_grid = {'learning_rate':np.linspace(1e-3, 1e1, 20) }



adboost = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),

                             n_estimators=100),

                        param_grid, n_jobs=2, cv=5)

adboost.fit(X_train, y_train)

print('Adaboost score: {}'.format(log_reg.score(X_train, y_train)))
voting = VotingClassifier([('log_reg',log_reg), ('decisiontree',tree),

                           ('randomforest',forest), ('adaboost',adboost)],

                          voting='soft') 

voting.fit(X_train, y_train)

voting.score(X_train, y_train)
X_test['Fare'] = X_test['Fare'].fillna(value=X_test['Fare'].mean())
X_test['Survived'] = forest.predict(X_test)

submission = X_test['Survived']




# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "forest.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

# create a link to download the dataframe

create_download_link(submission)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 


