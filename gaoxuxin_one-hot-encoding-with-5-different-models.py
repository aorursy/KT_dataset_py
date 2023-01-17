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

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

import statistics

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train.head())

print(test.head())

train = train.drop('Name',axis = 1)

test = test.drop('Name', axis = 1)
for col in ['PassengerId','Pclass','SibSp','Parch','Ticket','Cabin','Embarked']:

    

    print(train[col].unique())

    print(train[col].isnull().sum())

    print(len(train))

#ticket and cabin is usesless, remove them, the rest do one hot encoding

train.isnull().sum()

test.isnull().sum()

train['Embarked'].fillna(9,inplace = True)

train['Age'].fillna(train['Age'].median(),inplace = True)

test['Age'].fillna(test['Age'].median(),inplace = True)

test['Fare'].fillna(test['Fare'].median(),inplace = True)

scaler = StandardScaler()

train['Age'] = scaler.fit_transform(train['Age'].values.reshape(-1,1))

train['Fare'] = scaler.fit_transform(train['Fare'].values.reshape(-1,1))

test['Age'] = scaler.fit_transform(test['Age'].values.reshape(-1,1))

test['Fare'] = scaler.fit_transform(test['Fare'].values.reshape(-1,1))

test_ID = test['PassengerId']

for col in ['PassengerId','Ticket','Cabin']:

    train = train.drop(col,axis = 1)

    test = test.drop(col, axis = 1)
for col in ['Pclass','Sex','SibSp','Parch','Embarked']:

    one_hot_train = pd.get_dummies(train[col])

    one_hot_test = pd.get_dummies(test[col])

    for col_name in one_hot_train.columns:

        one_hot_train.rename(columns = {col_name : col+str(col_name)}, inplace = True)

        one_hot_test.rename(columns = {col_name : col+str(col_name)}, inplace = True)

    train = train.join(one_hot_train)

    test = test.join(one_hot_test)

    train = train.drop(col,axis = 1)

    test = test.drop(col,axis = 1)



#test = test.drop(9,axis = 1)

x = train.drop('Survived', axis = 1)

y = train['Survived']

data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(x, y, train_size = 0.8, random_state = 666)
def get_para_grid(name):

    graph_name.append(name)

    if name == 'RandomForest':

        para_grid = {'max_features': ['auto', 'sqrt'],

                     'min_samples_split': [2, 3, 4, 5, 6, 7, 10, 15]}

        clf = RandomForestClassifier()

        return (para_grid,clf)



    if name == 'DecisionTree':

        para_grid = {'max_leaf_nodes': list(range(5, 60)),

                     'min_samples_split': [2,3,4,5,6]}

        clf = DecisionTreeClassifier()

        return (para_grid,clf)



    if name == 'Logistic':

        para_grid = {'penalty': ['l1', 'l2'],

                     'C': list(np.logspace(-4, 4, 20)),

                     'solver': ['liblinear']}

        clf = LogisticRegression()

        return (para_grid,clf)



    if name == 'NB':

        para_grid = {'var_smoothing': list(np.logspace(0,-9, num=100))}

        clf = GaussianNB()

        return (para_grid,clf)



    if name == 'XGBoost':

        para_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8,1],

                     'max_depth':[3,5,8,10,20],

                     'max_features':["log2","sqrt"],

                     'subsample':[0.5, 0.8, 0.9, 1.0],

                     'n_estimators': [10]}

        clf = GradientBoostingClassifier()

        return(para_grid,clf)



def grid_search(data_train_x,data_test_x, data_train_y, data_test_y, para_grid,clf_name):

    CV = GridSearchCV(clf_name, para_grid,cv = 5)

    CV.fit(data_train_x, data_train_y)

    data_predict_y = CV.predict(data_test_x)

    accuracy = accuracy_score(data_test_y, data_predict_y)

    graph.append(accuracy)

    print('Model used: ',clf_name)

    print('Accuracy score is:', accuracy)

    print('Optimal parameter is:', CV.best_params_,'\n')

    global acc

    if accuracy > acc:

        acc = accuracy

        result = CV.predict(test)

        output = pd.DataFrame({'PassengerId':test_ID,'Survived':result})

        output.to_csv('my_submission.csv',index = False)
acc = 0.7
graph = []

graph_name = []

for m in ['RandomForest', 'DecisionTree', 'Logistic', 'NB', 'XGBoost']:

    grid,clf = get_para_grid(m)

    grid_search(data_train_x, data_test_x, data_train_y, data_test_y, grid, clf)

    

graph_name = tuple(graph_name)

plt.bar(np.arange(len(graph_name)),graph,color = ['black', 'red', 'green', 'blue', 'cyan'])

plt.xticks(np.arange(len(graph_name)),graph_name)

plt.show()