import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']

PassengerId

train.head(3)

train.info()
full_data=[train,test]



for f in full_data:

    f['Name_length']=f['Name'].apply(len)

    f['Has Cabin']=f['Cabin'].apply(lambda x:0 if type(x)==float else 1)

#sina boss add the family size to the feature list 

for dataset in full_data:

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1



for dataset in full_data:

    dataset['IsAlone']=dataset['FamilySize'].apply(lambda x:1 if x==1 else 0)

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#Fil the fare values with median 

for dataset in full_data:

    dataset['Fare']=dataset['Fare'].fillna(train['Fare'].median())



#fill the age values , if large number of ages arae there, then geenrate numbers between mean and std



for dataset in full_data:

    age_avg=dataset['Age'].mean()

    age_std=dataset['Age'].std()

    age_null_count= dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list

    dataset['Age']=dataset['Age'].astype(int)



train['CategoricalAge'] = pd.cut(train['Age'], 5)

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



#mapping the variables in discrete values now 

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

train = train.drop(['CategoricalAge'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

train.shape

test.shape
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train, hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits= NFOLDS, random_state=SEED)
print(train.shape)

print(test.shape)

def get_oof(clf,x_train,y_train,x_test):

    #contains the rows equal to no of training examples 

    oof_train=np.zeros((ntrain,))

    #contains the rows equal to no of testing examples

    oof_test=np.zeros((ntest,))

    #contains the nfolds rows, for each fold contains the predictions for the jth training examples(ntest in number)

    oof_test_skf=np.empty((NFOLDS,ntest))

    for i ,(train_index,test_index) in enumerate(kf.split(train)):

        #we get the total i folds,train and test indec ranges,now train on train_index range, store predictions of test_index,and evaluate

        #make predictiosn on the x_test

        x_tr=x_train[train_index]

        y_tr=y_train[train_index]

        x_te=x_train[test_index]

        #now fit the classifier on this fold 

        clf.train(x_tr,y_tr)

        #predict the values on x_te

        #each training example belongs to the test set once in folding so we can store the predictions in oof_train 

        oof_train[test_index]=clf.predict(x_te)

        #also store the predictions of this fold on the actual test set x_test

        oof_test_skf[i,:]=clf.predict(x_test)

    #when the loop ends, oof_train contains predictions whenever a portion of training set acts as a test set 

    #oof_test_skf contains the predictions for each kfold, for each test set.

    

    #next we need to average the predictions made for a particular test set across all kfolds. This means averaging the column values

    oof_test[:]=oof_test_skf.mean(axis=0)

    #we roll the vetors in the form of columns and return them 

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

        

        
for i, (train_index, test_index) in enumerate(kf.split(train)):

    print(i,train_index,test_index)
#we create the stacks of 5 models in first levels

#configure the parameters for each of them here 

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

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
rf=SklearnHelper(clf=RandomForestClassifier,seed=SEED,params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
#next create the train and test dataset for dfeeding to thte models 

y_train=train['Survived'].ravel()

train=train.drop(['Survived'],axis=1)

#remove the comlumns from pandas dataframes

x_train=train.values

x_test=test.values
#feed these datas to the ensemple and get the predictions

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

#next we evaluate the feature importances on the dataset 

rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371

  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]

et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157

  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]

ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008

  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]

gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854

  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]
#let us plot these features for different examples 

cols=train.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

      'AdaBoost feature importances': ada_features,

    'Gradient Boost feature importances': gb_features

    })

feature_dataframe
sns.catplot(y='Random Forest feature importances',x='features',data=feature_dataframe,height=5, aspect=4,)
sns.catplot(y='Extra Trees  feature importances',x='features',data=feature_dataframe,height=5, aspect=4,)
sns.catplot(y='AdaBoost feature importances',x='features',data=feature_dataframe,height=5, aspect=4,)

sns.catplot(y='Gradient Boost feature importances',x='features',data=feature_dataframe,height=5, aspect=4,)
#let us plot the mean of the rows for our dataset 

feature_dataframe['mean']=feature_dataframe.mean(axis=1)

feature_dataframe
#now let us plot the barplot of importance of the faetures 

sns.catplot(x="features", y="mean", kind="bar", data=feature_dataframe,aspect=4);
base_predictions_train=pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()                                     

    })



sns.heatmap(base_predictions_train.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap="YlGnBu", linecolor='white', annot=True)
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
print(x_train.shape)

print(x_test.shape)

gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)