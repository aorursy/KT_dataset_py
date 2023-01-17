# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import sklearn as sk

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold;

from sklearn.cross_validation import train_test_split
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')



PassengerId = test['PassengerId']
train_org = train

test_org = test

train.head(5)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, model_name, seed=0, params=None):

        if seed != None:

            params['random_state'] = seed

        self.clf = clf(**params)

        self.name = model_name



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def predict_proba(self, x):

        return self.clf.predict_proba(x)

    

    def score(self, x, y):

        return self.clf.score(x, y)

    

    def fit(self,x,y):

        try:

            return self.clf.fit(x,y)

        except AttributeError:

            return self.clf.train(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

        

    def model_name(self):

        return self.name

    

# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))

    train_accuracy = 0

    test_accuracy = 0



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        y_te = y_train[test_index]



        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict_proba(x_te)[:, 0]

        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 0]

        train_accuracy += clf.score(x_tr, y_tr)

        test_accuracy += clf.score(x_te, y_te)

    

    train_accuracy = train_accuracy/len(kf)

    test_accuracy = test_accuracy/len(kf)

    print('模型%s训练准确率为%f'%(clf.model_name(), train_accuracy))

    print('模型%s测试准确率为%f'%(clf.model_name(), test_accuracy))

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 3,

    'subsample':0.5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'poly',

    'C' : 0.025 ,

    'probability' : True

    }

gbm_params = {

    'learning_rate' : 0.4,

    'n_estimators' : 500,

    'max_depth' : 4,

    'min_child_weight': 2,

    #gamma=1,

    'gamma':0.9,                        

    'subsample':0.5,

    'colsample_bytree':0.8,

    'objective': 'binary:logistic',

    'reg_lambda':5,

    'nthread':-1,

    'scale_pos_weight' :1

}

lglr_params = {

    'C' : 0.5,

    'max_iter' : 1000

}

knn_params = {

    'n_neighbors' : 10,

    'weights' : 'uniform'

}
rf = SklearnHelper(RandomForestClassifier, 'RandomForest', seed=SEED, params=rf_params) # 

et = SklearnHelper(ExtraTreesClassifier, 'ExtraTrees',seed=SEED, params=et_params)

ada = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada_params)

gb = SklearnHelper(GradientBoostingClassifier, 'GradientBoosting', seed=SEED, params=gb_params)

svc = SklearnHelper(SVC, 'SVM',seed=SEED, params=svc_params)

gbm = SklearnHelper(xgb.XGBClassifier, 'XGB', seed=SEED, params=gbm_params)

lglr = SklearnHelper(sk.linear_model.LogisticRegression, 'logistic', seed=SEED, params=lglr_params)

knn = SklearnHelper(KNeighborsClassifier, 'KNN', seed=None, params=knn_params)
try:

    y_train = train['Survived'].ravel()

    train = train.drop(['Survived'], axis=1)

except KeyError:

    print('no need')

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

gbm_oof_train, gbm_oof_test = get_oof(gbm, x_train, y_train, x_test)

lglr_oof_train, lglr_oof_test = get_oof(lglr, x_train, y_train, x_test)

knn_oof_train, knn_oof_test= get_oof(knn, x_train, y_train, x_test)



print("Training is complete")
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel(),

       'SVC' : svc_oof_train.ravel(),

     'gbm' :gbm_oof_train.ravel(),

    'lglr' : lglr_oof_train.ravel(),

    'knn' : knn_oof_train.ravel()

    })

base_predictions_train.head()
sns.heatmap(base_predictions_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,  linecolor='white', annot=True)
sec_x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, svc_oof_train, lglr_oof_train,knn_oof_train), axis=1)

sec_x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, svc_oof_test,lglr_oof_test,knn_oof_test), axis=1)
gbm2_params = {

    'learning_rate' : 0.2,

    'n_estimators' : 500,

    'max_depth' : 4,

    'min_child_weight': 3,

    #gamma=1,

    'gamma':0.9,                        

    'subsample':0.4,

    'colsample_bytree':0.8,

    'objective': 'binary:logistic',

    'reg_lambda':2,

    'nthread':-1,

    'scale_pos_weight' :1

}

lglr2_params = {

     'C' : 0.5,

    'max_iter' : 1000

}

svc2_params = {

    'kernel' : 'linear',

    'C' : 0.025 ,

    'probability' : True

}

ada2_params = {

    'n_estimators': 30,

    'learning_rate' : 0.9

}
gbm2 = SklearnHelper(xgb.XGBClassifier, 'XGB', seed=SEED, params=gbm2_params)

lglr2 = SklearnHelper(sk.linear_model.LogisticRegression, 'lglr', seed=SEED, params=lglr2_params)

svc2 = SklearnHelper(SVC, 'svc', seed=SEED, params=svc2_params)

adaboost2 = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada2_params)
x_train2, x_dev2, y_train2,y_dev2 = train_test_split(sec_x_train, y_train, test_size = 0.2)
_,_ = get_oof(gbm2, sec_x_train, y_train, sec_x_test)

_,_ = get_oof(lglr2, sec_x_train, y_train, sec_x_test)

_,_ = get_oof(svc2, sec_x_train, y_train, sec_x_test)

_,_ = get_oof(adaboost2, sec_x_train, y_train, sec_x_test)
#adaboost2.fit(x_train2, y_train2)

print ("在训练集上的准确率为%f"%(adaboost2.score(x_train2, y_train2)))

print ("在测试集上的准确率为%f"%(adaboost2.score(x_dev2, y_dev2)))
num_tree = [1,2, 5, 10, 15, 20, 30 , 40, 50, 100, 150 ,200, 300,500]

train_accuracy = []

test_accuracy = []

ada2_params['learning_rate'] = 1

for num in num_tree:

    ada2_params['n_estimators'] = num

    adaboost2 = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada2_params)

    adaboost2.fit(x_train2, y_train2)

    train_accuracy.append(adaboost2.score(x_train2, y_train2))

    test_accuracy.append(adaboost2.score(x_dev2, y_dev2))

plt.plot(num_tree, train_accuracy, 'r')

plt.plot(num_tree, test_accuracy, 'b')

plt.show()
models = [gbm2, lglr2, svc2, adaboost2]

for model in models:

    predictions = model.predict(sec_x_test)

    Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

    Submission.to_csv(model.name + ".csv", index = False)