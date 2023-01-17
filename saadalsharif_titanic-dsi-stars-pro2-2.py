# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from scipy.stats import norm

import warnings

import datetime

import time

# Importing libraries for Modeling

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

plt.style.use('ggplot')

sns.set(font_scale=1.5)

%config InlineBackend.figure_format = 'retina'

%matplotlib inline
#import data file for kaggle

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#import data file for local work

#train = pd.read_csv('../data/train.csv')

#test = pd.read_csv('../data/test.csv')
train.head()
test.head()
train.shape, test.shape
train.info()
test.info()
train.select_dtypes(include=object).head()
test.select_dtypes(include=object).head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



train.select_dtypes(include=numerics)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



test.select_dtypes(include=numerics)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train.isnull().sum()
test.isnull().sum()
train.Embarked.value_counts()
x =pd.isnull(test['Fare'])

test[x]
mean_Fare = test.groupby('Pclass')['Fare'].mean()

mean_Fare

print("The mean fare for the Pclass (for missing fare data) is: {}".format(mean_Fare[3]))
cc =test['Fare'].replace(np.nan , mean_Fare[3], inplace=True )
mean_age = train.groupby('Pclass')[['Age']].mean()

mean_age
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
train.Age = train.apply(lambda x :impute_age(x[['Age', 'Pclass']] ) , axis = 1)



test.Age = test.apply(lambda x :impute_age(x[['Age', 'Pclass']] ) , axis = 1)
test['Cabin']= test['Cabin'].apply(lambda x :0 if pd.isnull(x)else 1)

train['Cabin']=train['Cabin'].notnull().astype('int')
train.Embarked.value_counts()
train.Embarked.replace(np.nan ,'S', inplace= True )
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
survivors = train[train['Survived'] == 1]

train['Survived'].value_counts(normalize=True)
train.groupby(['Sex','Survived']).size().reset_index(name='Frequency')
pd.crosstab(train['Sex'],train['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1)
train.groupby(['Pclass','Survived']).size().reset_index(name='Frequency')
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab(train['Pclass'],train['Survived'],margins=True).style.background_gradient(cmap='PuBu')
pd.crosstab(train['Pclass'],train['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1)
pd.crosstab(train['Embarked'],train['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1).tail()
age_less_12 = train[ train['Age'] < 12 ]

pd.crosstab(age_less_12['Pclass'],age_less_12['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1).tail()
print('Oldest Passenger was of:',round(train['Age'].max()),'Years')

print('Youngest Passenger was of:',round(train['Age'].min(),1),'Years')

print('Average Age on the ship:',round (train['Age'].mean()),'Years')
train[ train['Name'].str.contains('Cap') ]
fig,ax=plt.subplots(1,2,figsize=(18,8))



train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)



ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Sex : Survived vs Dead')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train['Pclass'].value_counts().plot.bar(color=['#CD7F33','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
train.head(3)
test.head(3)
selected_features = ['PassengerId','Pclass','Age', 'SibSp', 'Parch',

                     'Fare','Cabin','Sex_male', 'Embarked_Q',

                     'Embarked_S']

selected_features
X = train[selected_features]

y = train['Survived']



X_test  = test.drop(["Name",'Ticket'], axis=1).copy()
X.shape,X_test.shape
set(X_test.columns).symmetric_difference(set(X.columns))
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X, y)





sgd.score(X, y)



acc_sgd = round(sgd.score(X, y) * 100, 2)
cv = StratifiedKFold(n_splits=10, shuffle=True)

scaler = StandardScaler()

forest = RandomForestClassifier()

forest_pipeline = Pipeline([('transformer', scaler), ('estimator', forest)])



forest_params = {'estimator__n_estimators': [5,50,80,100],

              'estimator__max_depth':[1,2,3,4,5,6,7],

                'estimator__max_features':[2,5,7,9]}



forest_grid = GridSearchCV(forest_pipeline, forest_params,

                           n_jobs=-1, cv=cv, verbose=2)

forest_grid.fit(X, y);

best_forest = forest_grid.best_estimator_

print(f' GridSearch best score: {forest_grid.best_score_}')

print(f' GridSearch best params : {forest_grid.best_params_}')

cv = StratifiedKFold(n_splits=10, shuffle=True)

scaler = StandardScaler()

ada = AdaBoostClassifier()

ada_pipeline = Pipeline([('transformer', scaler), ('estimator', ada)])



ada_params = {'estimator__base_estimator': [None, DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)],

              'estimator__n_estimators': [10,50, 70],

              'estimator__learning_rate':[.01, .1, .5, 1]}



ada_grid = GridSearchCV(ada_pipeline, ada_params, n_jobs=-1, cv=cv, verbose=2)

ada_grid.fit(X, y);

best_ada = ada_grid.best_estimator_

print(f' GridSearch best score: {ada_grid.best_score_}')

print(f' GridSearch best params : {ada_grid.best_params_}')
#KNN to gid search CV



n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = list(range(1,50,5))

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,n_jobs=-1,

                cv=10, scoring = "roc_auc")

gd.fit(X, y)

best_knn = ada_grid.best_estimator_

print(gd.best_score_)

print(gd.best_estimator_)
submit = pd.DataFrame({'PassengerId':X_test.PassengerId, 

                    'Survived':best_knn.predict(X_test).astype(int)})





submit.to_csv("gender_submission.csv", index=False)
submit.head(5)
submit.tail(5)