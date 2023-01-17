#loading all the required packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#loading the train and test dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_raw = train.copy()

test_raw = test.copy()

print("Train : {}".format(train.shape))

print("Test : {}".format(test.shape))
train.head()
test.head()
#some useful information about the data

train.info()
test.info()
#dropping PassengerId columns from both train and test datasets

train.drop(columns = ['PassengerId'],inplace = True)

test.drop(columns = ['PassengerId'],inplace = True)
train['Pclass'] = train['Pclass'].astype(str)

test['Pclass'] = test['Pclass'].astype(str)
for dataset in (train,test) :

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)

    #dropping Name column

    dataset.drop(columns = ['Name'],inplace = True)
train['Title'].groupby(by = train['Title']).count()
test['Title'].groupby(by = test['Title']).count()
#aggregating titles

Title_Dict = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "Countess": "Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"



                    }

train['Title'] = train['Title'].map(Title_Dict)

test['Title'] = test['Title'].map(Title_Dict)
Age_dict = train.groupby(by = 'Title')['Age'].mean().astype(int).to_dict()

Age_dict
#filling the missing values

for dataset in (train,test):

    nan_idx = dataset.loc[dataset['Age'].isnull()].index

    dataset.loc[nan_idx,'Age'] = dataset.loc[nan_idx,'Title'].map(Age_dict)
#Creating Age bins

#Taking a look at the categories

#quantile based discretization

pd.qcut(train['Age'],q = 5).head()
#Let's create bins based on the above categories

bins = [0,20,26,32,38,80]

train['Age'] = pd.cut(train['Age'],bins = bins,

                      labels = ['Age_{}'.format(str(x)) for x in np.arange(1,6,1)])

test['Age'] = pd.cut(test['Age'],bins = bins,

                    labels = ['Age_{}'.format(str(x)) for x in np.arange(1,6,1)])
for dataset in (train,test):

    #Creating a feature called the Family size

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize

    dataset['IsAlone'] = (dataset['FamilySize'] == 1) * 1



train.drop(columns = ['SibSp','Parch'],inplace = True)

test.drop(columns = ['SibSp','Parch'],inplace = True)

#dropping ticket feature as ticket number doesn't help in predicting survival

train.drop(columns = ['Ticket'],inplace = True)

test.drop(columns = ['Ticket'],inplace = True)
test.loc[test['Fare'].isnull()]
test['Fare'].fillna(test.loc[test['Pclass'] == '3','Fare'].mean(),inplace = True)
#Creating fare bins

#Quantile cut

pd.qcut(train['Fare'],q = 4).head()
#Creating fare bins based on the above categories

fare_bins = [-0.001,7.91,14.454,31,513]

train['Fare'] = pd.cut(train['Fare'],bins = fare_bins,

                       labels = ['Fare_{}'.format(str(x)) for x in np.arange(1,5,1)])

test['Fare'] = pd.cut(test['Fare'],bins = fare_bins,

                     labels = ['Fare_{}'.format(str(x)) for x in np.arange(1,5,1)])
for dataset in (train,test):

    dataset['Cabin'].fillna('U',inplace = True)

    dataset['Cabin'] = dataset['Cabin'].apply(lambda x : x[0])
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace = True)
test.info()
train.info()
train = pd.get_dummies(train)

test = pd.get_dummies(test)
#separating out target variables and predictor variables

y_train = train['Survived']

#dropping Cabin_T column also as it is not present in the test dataset

train.drop(columns = ['Survived','Cabin_T'],inplace = True)
#defining our error metric

from sklearn.model_selection import StratifiedKFold,cross_val_score

def accuracy(model):

    skfold = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 66)

    acc = cross_val_score(model,X = train,y = y_train,scoring = 'accuracy',cv = skfold)

    return acc.mean()
#finding C in LogisticRegression model

#from sklearn.linear_model import LogisticRegression

#for c in [0.001,0.01,0.1,1,10,100]:

    #lr = LogisticRegression(penalty='l2',solver = 'lbfgs',C = c,random_state = 6,max_iter = 500)

    #print("{} : {:.4f}".format(c,accuracy(lr)))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2',solver = 'lbfgs',C = 0.1,random_state = 6,max_iter = 100)

print("Logistic Regression score : {:.4f}".format(accuracy(lr)))

lr.fit(train,y_train)

pred_lr = lr.predict(test)

submission_lr = pd.DataFrame({"PassengerId" : test_raw["PassengerId"], "Survived" : pred_lr })

submission_lr.to_csv("logistic_regression",index = False)
#cross validation to tune the parameters of random forest

#from sklearn.model_selection import GridSearchCV

#from sklearn.ensemble import RandomForestClassifier

#parameter_grid = {'n_estimators' : [10,50,100,200,500],

#                 'criterion' : ['entropy','gini'],

#                 'max_features' : ['log2', 'sqrt','auto'],

#                  'min_samples_leaf': [1,5,8],

#                 'max_depth': [50,80,90,100,110],

 #                 'min_samples_split':[2,3,5]

  #               }

#skfold = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 66)

#grid_search = GridSearchCV(RandomForestClassifier(random_state = 666),param_grid = parameter_grid,

 #                          scoring = 'accuracy',n_jobs = -1,iid = False,cv = skfold,verbose = 2 )

#grid_search.fit(train,y_train)

#print(grid_search.best_params_)

#print(grid_search.best_score_)
#Let's fit a random forest model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 50,max_depth = 90,criterion = 'entropy',

                            max_features = 'log2',min_samples_leaf = 5,random_state = 55,

                            min_samples_split = 2) #parameters estimated using cross validation

print("Random Forest score : {:.4f}".format(accuracy(rf)))

rf.fit(train,y_train)

pred_rf = rf.predict(test)

submission_rf = pd.DataFrame({"PassengerId" : test_raw["PassengerId"], "Survived" : pred_rf })

submission_rf.to_csv("random_forest",index = False)
#cross validation

# from sklearn.model_selection import GridSearchCV

# import xgboost as xgb

# param_grid = {#'n_estimators' : [10,100,200,300,400,500,700,1000],

#               #'max_depth' : [2,3,4,5],

#                # 'min_child_weight' : [1,2,3,4]

#                 #'gamma' : [0.0,0.1,0.2,0.3,0.4,0.5],

#                 #'colsample_bytree' : [0.6,0.7,0.8,0.9,1.0],

#                 #'subsample' : [0.6,0.7,0.8,0.9,1.0],

#                 #'reg_alpha' : [0.01,0.03,0.1,0.3,1,3,10,30],

#                 #'reg_lambda' :[0.01,0.03,0.1,0.3,1,3,10,30]

#              } 

# skfold = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 66)

# XGB = xgb.XGBClassifier(learning_rate = 0.05,n_jobs = -1,max_depth = 2,n_estimators = 200,

#                        subsample = 0.9,colsample_bytree = 0.9,min_child_weight = 1,

#                         gamma = 0.0,reg_alpha = 0.01,reg_lambda = 1,random_state = 66)

# grid_search = GridSearchCV(XGB,param_grid = param_grid,scoring = 'accuracy',

#                            n_jobs = -1,iid = False,cv = skfold,verbose = 2)

# grid_search.fit(train,y_train)

# print(grid_search.best_params_)

# print(grid_search.best_score_)



import xgboost as xgb

XGB = xgb.XGBClassifier(learning_rate = 0.05,n_jobs = -1,max_depth = 2,n_estimators = 200,

                       subsample = 0.9,colsample_bytree = 0.9,min_child_weight = 1,

                        gamma = 0.0,reg_alpha = 0.01,reg_lambda = 1,

                        random_state = 66)   #parameters found by cross validation

print("XGB score : {:.4f}".format(accuracy(XGB)))

XGB.fit(train,y_train)

pred_xgb = XGB.predict(test)

submission_xgb = pd.DataFrame({"PassengerId" : test_raw["PassengerId"], "Survived" : pred_xgb })

submission_xgb.to_csv("XGBoost",index = False)