# For data analysis and data wragling

import pandas as pd

import numpy as np



# For data visualization and graph plotting



import seaborn as sns

import plotly.graph_objs as go

import plotly.tools as tls

import matplotlib.pyplot as plt

%matplotlib inline



# For Machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import plotly.offline as py

py.init_notebook_mode(connected=True)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
dataset = pd.read_csv('/kaggle/input/train.csv')

dataset.head(4)
dataset.dtypes
print(dataset.describe())

print(dataset.Age.median())

print(dataset.Fare.median())
dataset[['Sex','Survived']].groupby('Sex',as_index=False).mean()
g = sns.FacetGrid(dataset, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(dataset, col='Survived')

g.map(plt.hist, 'Fare', bins=20)
dataset.Pclass.unique()
dataset[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()
dataset.SibSp.unique()
dataset[['SibSp' , 'Survived']].groupby('SibSp' , as_index = False).mean()
dataset.Parch.unique()
dataset[['Parch' , 'Survived']].groupby('Parch' , as_index = False).mean()
dataset.Embarked.unique()
dataset[['Embarked' , 'Survived']].groupby('Embarked' , as_index = False).mean()
grid = sns.FacetGrid(dataset, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(dataset, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
df = dataset[['Pclass','Sex','Age','Fare','Embarked','Survived', 'SibSp' , 'Parch']]

df.head()
df.isnull().sum()
df.fillna({'Age': df.Age.mean() ,

          'Embarked': 'S'} , inplace = True)
df.head()
df.isnull().sum()
df = pd.get_dummies(df , columns=['Sex' , 'Embarked' , 'Pclass' , 'Parch'])

print(df)
X = df.drop(columns='Survived').values

Y = df['Survived'].values

print(X)

print(Y)
scale = StandardScaler()

X[: , 0:2] = scale.fit_transform(X[:, 0:2])

print(X)
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)

print(X_train)
classifier = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=5, max_features='log2',

                       max_leaf_nodes=None, max_samples= None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=3, min_samples_split=8,

                       min_weight_fraction_leaf=0.0, n_estimators=300,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)
pred = cross_val_score(classifier , X , Y , cv = 3).mean()

print(pred)
classifier.fit(X_train , Y_train)
y_pred = classifier.predict(X_test)

print(y_pred)
plot_confusion_matrix(classifier , X_test , Y_test , display_labels=["Does not Survived" , "Survived"])
from sklearn.metrics import accuracy_score

score = accuracy_score(Y_test , y_pred)

print(score)
# from sklearn.model_selection import GridSearchCV 

  

# # defining parameter range 

# param_grid = {'n_estimators': [300,500],

#     'max_features': ['auto', 'sqrt', 'log2'],

#     'max_depth' : [5,6,8,80],

#     'min_samples_leaf': [3, 4, 5],

#     'min_samples_split': [8, 10],}  

  

# grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3) 

  

# # fitting the model for grid search 

# grid.fit(X_train, Y_train) 



# # print best parameter after tuning 

# print(grid.best_params_) 



# print(grid.best_estimator_) 
classifier_2 = RandomForestClassifier(n_jobs =  -1,

    n_estimators= 300,

     warm_start= True, 

#      max_features= 0.2,

    random_state= 2,

    max_depth= 5,

    min_samples_leaf= 3,

    max_features = 'log2',

    criterion= 'entropy',  

    min_samples_split= 8,

    verbose =  0)
predict = cross_val_score(classifier_2 , X , Y , cv = 3).mean()

print(predict)
classifier_2.fit(X_train , Y_train)
y_predict = classifier_2.predict(X_test)

print(y_predict)
plot_confusion_matrix(classifier_2 , X_test , Y_test , display_labels=["Does not Survived" , "Survived"])
from sklearn.metrics import accuracy_score

score_2 = accuracy_score(Y_test , y_predict)

print(score_2)
from sklearn.svm import SVC



# # {'C': 1, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf', 'random_state': 2}

# SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,

#     decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',

#     max_iter=-1, probability=False, random_state=2, shrinking=True, tol=0.001,

#     verbose=False)



classifier_3 = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',

    max_iter=-1, probability=False, random_state=2, shrinking=True, tol=0.001,

    verbose=False)

print(classifier_3)
predict_2 = cross_val_score(classifier_3 , X , Y , cv = 3).mean()

print(predict_2)
classifier_3.fit(X_train , Y_train)
y_prediction = classifier_3.predict(X_test)

print(y_prediction)
plot_confusion_matrix(classifier_3 , X_test , Y_test , display_labels=["Does not survived" , "Survived"])
from sklearn.metrics import accuracy_score

score_3 = accuracy_score(Y_test , y_prediction)

print(score_3)
# from sklearn.model_selection import GridSearchCV 

  

# # defining parameter range 

# param_grid = {'C': [0.1, 1, 10, 100, 1000],  

#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

#                'degree': [ 2, 3 , 4 ,5],

#               'random_state': [ 2, 3, 4 , 5],

#               'kernel': ['rbf']}  

  

# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

  

# # fitting the model for grid search 

# grid.fit(X_train, Y_train) 



# # print best parameter after tuning 

# print(grid.best_params_) 



# print(grid.best_estimator_) 
from sklearn.linear_model import LogisticRegression





# {'C': 0.09, 'penalty': 'l2', 'random_state': 2}

# LogisticRegression(C=0.09, class_weight=None, dual=False, fit_intercept=True,

#                    intercept_scaling=1, l1_ratio=None, max_iter=100,

#                    multi_class='auto', n_jobs=None, penalty='l2',

#                    random_state=2, solver='lbfgs', tol=0.0001, verbose=0,

#                    warm_start=False)





classifier_4 = LogisticRegression(C=0.09, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=2, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)

print(classifier_4)
predict_4 = cross_val_score(classifier_4 , X , Y , cv = 3).mean()

print(predict_4)
classifier_4.fit(X_train , Y_train)
y_prediction_lr = classifier_4.predict(X_test)

print(y_prediction_lr)
plot_confusion_matrix(classifier_4 , X_test , Y_test , display_labels=["Does not survived" , "Survived"])
from sklearn.metrics import accuracy_score

score_lr = accuracy_score(Y_test , y_prediction_lr)

print(score_lr)
# from sklearn.model_selection import GridSearchCV 

  

# # defining parameter range 

# param_grid = {'penalty': ['l1', 'l2'],

#               'C':[0.001,.009,0.01,.09,1,5,10,25],

#               'random_state': [2 , 3 , 4]}  

  

# grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 3) 

  

# # fitting the model for grid search 

# grid.fit(X_train, Y_train) 



# # print best parameter after tuning 

# print(grid.best_params_) 



# print(grid.best_estimator_) 