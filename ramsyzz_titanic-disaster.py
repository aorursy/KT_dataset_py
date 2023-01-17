# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=0)

training['train_data'] = 1
test['test_data'] = 0
test['Survived'] = np.NaN # willn't consider survived data into test set
data = pd.concat([training,test])

data.columns
training.head()
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import sklearn
#from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import sklearn.neighbors as neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
training.info()
training.describe().columns
# Percentage of missing values in dataframe
training.isnull().sum().sort_values(ascending=False)
(training.isnull().sum()/training.isnull().sum().count()).sort_values(ascending=False)
# numeric and categorical var
df_num = training[['Age', 'SibSp','Parch','Fare']]
df_cat = training[['Survived', 'Pclass','Sex', 'Ticket','Cabin','Embarked']]

print(df_num.corr())
sns.heatmap(df_num.corr())
# survived percentage rate across Age, SibSp, Parch and Fare ( numeric var)
pd.pivot_table(training, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare'])

for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
    plt.show()
print(pd.pivot_table(training, index = 'Survived', values = 'Ticket', columns = 'Pclass', aggfunc= 'count'))

print(pd.pivot_table(training, index = 'Survived', values = 'Ticket', columns = 'Sex', aggfunc= 'count'))

print(pd.pivot_table(training, index = 'Survived', values = 'Ticket', columns = 'Embarked', aggfunc= 'count'))
#df_cat.cabin
training['cabin'] = training.Cabin.apply(lambda x : str(x)[0])
training['cabin'].value_counts()
pd.pivot_table(training, index = 'Survived', values = 'Name', columns = 'Cabin', aggfunc = 'count')
# Ticket variable splitting from letters n numeric 
training['numeric_tickets'] = training.Ticket.apply(lambda x : 1 if x.isnumeric() else 0)
training['ticket_letters'] = training.Ticket.apply(lambda x : ''.join(x.split(' ')[:-1]).replace
                                        ('.','').replace('/','').lower() if len(x.split(' ')[:-1])>0  else 0)

training['numeric_tickets'].value_counts()
pd.pivot_table(training, index = 'Survived', values = 'Ticket', columns = 'numeric_tickets', aggfunc = 'count')
# Suffix name(Title of person) splitting from first & last name using titles of person for exploration
training.Name.head()
training['name_title'] = training.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())

training['name_title'].value_counts()
# Age  [impute nulls(might be 0,1,-1) continuous var]
training['Age'] = training.Age.fillna(training.Age.mean())
training['Age'].head()
# Fare [impute nulls(might be 0,1,-1) continuous var]
training['fare'] = training.Fare.fillna(training.Age.median())
training['fare'].head()
training.head()
#  Process data for model ( Above preaprtion now creating all categorical var for train and test data )
# Concating all data

data['Cabin'] = data.Cabin.apply(lambda x : str(x)[0])
data['numeric_tickets'] = data.Ticket.apply(lambda x : 1 if x.isnumeric() else 0)
data['ticket_letters'] = data.Ticket.apply(lambda x : ''.join(x.split(' ')[:-1]).replace
                                        ('.','').replace('/','').lower() if len(x.split(' ')[:-1])>0  else 0)
data['name_title'] = data.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())


# drop observations without Embarked rows(have 2 nulls)
data.dropna(subset = ['Embarked'], inplace = True) 

data['Age'] = data.Age.fillna(training.Age.mean())
data['fare'] = data.Fare.fillna(training.Age.median())


# Transform categorical features into dummy variables
data_dummies = pd.get_dummies(data[['Pclass','Sex','SibSp','Parch','Cabin','numeric_tickets','name_title','Embarked','Age','fare','train_data']])

# split the train from train_test
X_train = data_dummies[data_dummies.train_data == 1].drop(['train_data'], axis = 1)
X_test = data_dummies[data_dummies.train_data == 0].drop(['train_data'], axis = 1)

y_train = data[data.train_data==1].Survived

y_train.shape
X_train
# Debug
print('Inputs: \n', X_train.head(2))

print('Outputs: \n', y_train.head(2))
# Standardize features by removing the mean and scaling to unit variance

from sklearn.preprocessing import StandardScaler

scale = StandardScaler() # scale data
data_dummies_scaled  = data_dummies.copy()
data_dummies_scaled[['Age','SibSp','Parch','fare']] = scale.fit_transform(data_dummies_scaled[['Age','SibSp','Parch','fare']])
data_dummies_scaled

X_train_scaled = data_dummies_scaled[data_dummies_scaled.train_data == 1].drop(['train_data'], axis = 1)
X_test_scaled = data_dummies_scaled[data_dummies_scaled.train_data == 0].drop(['train_data'], axis = 1)

y_train = data[data.train_data==1].Survived



from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
gnb = GaussianNB()
scores = cross_val_score(gnb, X_train_scaled, y_train, cv=5)
print(scores)
print(scores.mean())
logreg = LogisticRegression(max_iter=1000)
scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
print(scores)
print(scores.mean())


neigh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(neigh, X_train_scaled, y_train, cv=5)
print(scores)
print(scores.mean())
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Bestscore:' +str(classifier.best_score_))
    print('BestParameters:' +str(classifier.best_params_))
logreg = LogisticRegression()
param_grid = {'max_iter': [2000],
             'penalty': ['l1','l2'],
              'C': np.logspace(-4,4,20),
              'solver': ['liblinear']}
clf_logreg = GridSearchCV(logreg, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_logreg = clf_logreg.fit(X_train_scaled, y_train)
clf_performance(best_clf_logreg, 'Logistic Regression')
neigh = KNeighborsClassifier(n_neighbors=3)


param_grid = {'n_neighbors' : [3,5,7,9],
             'weights': ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree', 'kd_tree'],
              'p': [1,2]}
clf_neigh = GridSearchCV(neigh, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_neigh = clf_neigh.fit(X_train_scaled, y_train)
clf_performance(best_clf_neigh, 'KNN')
#XGBoost Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state = 1)
param_grid = {
    'n_estimators' : [350, 400, 450],
    'max_depth' : [None],
    'reg_alpha' : [1],
    'reg_lambda' : [2,5,10],
    'subsample' :  [0.55,0.6],
    'colsample_bytree' : [.75,.8,.85],
    'learning_rate' : [0.5],
    'gamma': [.5,1,2],
    'min_child_weight' : [0.01],
    'sampling_method' : ['uniform']}
clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train_scaled, y_train)
clf_performance(best_clf_xgb, 'XGB')

