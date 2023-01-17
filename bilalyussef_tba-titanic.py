# General Libraries

import pandas as pd

import numpy as np

import random as rnd

from scipy import stats

from collections import Counter



# Cleaning and preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split



# visiualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.style as style

import matplotlib.gridspec as gridspec

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

'''Customize visualization.'''

plt.style.use('bmh')                    # Use bmh's style for plotting

sns.set_style({'axes.grid':False})      # Remove gridlines

'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))



# modeling 



# Classification libraries and evaluation

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import metrics #accuracy measure



# Regression Libraries

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet, RidgeCV, LassoCV, ElasticNetCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error



# Other models

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn import svm 



# To ignore unwanted warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train.Embarked.value_counts()
train.Embarked.fillna('S',inplace= True)
ME = train[train['Pclass'] == 3]['Fare'].mean()

test.Fare.fillna(ME,inplace=True)
train.groupby('Pclass').mean()[['Age']]
#defining a function 'impute_age'

def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass']



    # Passing age_pclass[0] which is 'Age' to variable 'Age'

    Age = age_pclass[0]

    

    # Passing age_pclass[2] which is 'Pclass' to variable 'Pclass'

    Pclass = age_pclass[1]

    

    #applying condition based on the Age and filling the missing data respectively 

    if pd.isnull(Age):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 30



        else:

            return 25



    else:

        return Age
# (for train) grab age and apply the impute_age, our custom function

train['Age'] = train.apply(lambda x: impute_age(x[['Age','Pclass']]), axis=1)

# (for test) grab age and apply the impute_age, our custom function 

test['Age'] = test.apply(lambda x: impute_age(x[['Age','Pclass']]), axis=1)

train.isnull().sum()
test.isnull().sum()
train['Cabin'] = train['Cabin'].apply(lambda x: 1 if pd.isnull(x) else 0)

train.isnull().sum()
test['Cabin'] = test['Cabin'].apply(lambda x: 1 if pd.isnull(x) else 0)

test.isnull().sum()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train = pd.get_dummies(train, columns=['Sex','Embarked'],drop_first=True)

train
test = pd.get_dummies(test, columns=['Sex','Embarked'],drop_first=True)

test
train
X_train = train[['Pclass','Age','SibSp','Parch','Fare','Cabin','Sex_male','Embarked_Q','Embarked_S']]

y_train = train['Survived']

X_test = test[['Pclass','Age','SibSp','Parch','Fare','Cabin','Sex_male','Embarked_Q','Embarked_S']]



knn = KNeighborsClassifier()

knn_cv_accuracy =cross_val_score(knn, X_train, y_train, cv=5).mean()

print('Mean cross-validated accuracy for default knn:',knn_cv_accuracy)
knn_params = {

    'n_neighbors': [5,9,15,25,40,50,60],

    'weights':['uniform','distance'],

    'metric':['euclidean','manhattan']}

print('Initialized parameters for Grid Search')

print(knn_params)
knn_gridsearch = GridSearchCV(KNeighborsClassifier(), 

                              knn_params, 

                              n_jobs=-1, cv=5)

knn_gridsearch.fit(X_train, y_train)



knn_gridsearch.best_score_
knn_gridsearch.best_params_
y_test = pd.read_csv('../input/titanic/gender_submission.csv')

y_test.drop(columns= 'PassengerId',inplace=True)
y_test.columns
best_knn = knn_gridsearch.best_estimator_

best_knn.score(X_test, y_test)

y_test = best_knn.predict(X_test)
test['Survived'] = y_test
save_file = test[['PassengerId','Survived']]
save_file.to_csv('save_file.csv',index=False)
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)
y_rf = rf.predict(X_test)

rf.score(X_test,y_test)
test['Survived'] = y_rf

save_file = test[['PassengerId','Survived']]

save_file.to_csv('save_file.csv',index=False)
et = ExtraTreesClassifier(n_estimators=100) # bootstrap=False by default #max_features='auto',

et.fit(X_train,y_train)

y_et = rf.predict(X_test)

test['Survived'] = y_et

save_file = test[['PassengerId','Survived']]

save_file.to_csv('save_file.csv',index=False)
y_et
y_rf
rf_params = {

   'n_estimators': [10, 50, 100, 150, 200, 250],

   'max_features':[2, 3, 5, 7, 8],

    'max_depth': [1, 2, 3, 4, 5, 8],

   'criterion':['gini', 'entropy'],

}

rf_g = RandomForestClassifier(n_estimators=100)

gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1,n_jobs=-1)



gs.fit(X_train,y_train)

gs.score(X_train,y_train)
y_rf_g = gs.predict(X_test)

test['Survived'] = y_rf_g

save_file = test[['PassengerId','Survived']]

save_file.to_csv('save_file.csv',index=False)