import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
Fulldata = [data_train, data_test]
data_train.info()
# Step 0 : Filling missing values 

for data in Fulldata :

    data['Age'] = data['Age'].fillna(-0.5)

    data['Fare'] = data['Fare'].fillna(-0.5)

    data['Cabin'] = data['Cabin'].fillna('N')

    data['Embarked'] = data['Embarked'].fillna('N')



# Step 1 : Simplifying some features 

# Simplify Age : 

for data in Fulldata:

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(data['Age'], bins, labels=group_names)

    data['Age'] = categories

# Simplify Fares :

for data in Fulldata:

    data['Fare'] = data['Fare'].astype('float')

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(data['Fare'], bins, labels=group_names)

    data['Fare'] = categories

# Split Cabins into CabinFloor & CabinNumber :    

for data in Fulldata:

    a = data['Cabin'].str.extract('([A-Z])([0-9]+)', expand = False) #check that we fix the problem 'B12 B14 B16' 

    data['CabinFloor'] = a[0]

    data['CabinNumber'] = a[1]

    data['CabinFloor'] = data['CabinFloor'].fillna('N')

# Simplify CabinNumber :

    data['CabinNumber'] = data['CabinNumber'].astype('float')

    data['CabinNumber'] = data['CabinNumber'].fillna(-0.5)

    bins = (-1, 0, 74, 148)

    group_names =['Unknown','1st_side','2nd_side']

    categories = pd.cut(data['CabinNumber'], bins, labels=group_names)

    data['CabinNumber'] = categories

    

# Step 2 : Adding some new features : 

# Creating FamilySize and IsAlone : 

for dataset in Fulldata:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in Fulldata:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Now we extract infos from Name : 

## Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in Fulldata:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in Fulldata:

    dataset['Title'] = dataset['Title'].fillna('N')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Don', 'Dr', 'Rev', 'Sir', 'Jonkheer', 'Dona','Master'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Major'],'Spec')

    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','Mme','Mr','Miss','Mrs'], 'Std')
data_train.info()
#Step 4 : Map features into Numbers

for data in Fulldata:

# Mapping Sex

    data['Sex'] = data['Sex'].fillna(2)

    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Mapping Titles 

    title_map = {'Std':1, 'Rare':2,'Spec':3}

    data['Title']= data['Title'].fillna(0)

    data['Title']= data['Title'].map(title_map).astype(int)

# Mapping Embarked 

    data['Embarked'] = data['Embarked'].map({'N':0,'S':1,'C':2,'Q':3}).astype(int)

# Mapping Age

    age_map = {'Unknown':0, 'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}

    data['Age'] = data['Age'].map(age_map).astype(int)

# Mapping Fare

    fare_map = {'Unknown':0, '1_quartile':1, '2_quartile':2, '3_quartile':3, '4_quartile':4}

    data['Fare'] = data['Fare'].map(fare_map).astype(int)

# Mapping CabinFloor

    cfloor_map = {'N':0,'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}

    data['CabinFloor'] = data['CabinFloor'].map(cfloor_map).astype(int)

# Mapping CabinNumber

    cnumber_map = {'Unknown':0,'1st_side':1,'2nd_side':2}

    data['CabinNumber'] = data['CabinNumber'].map(cnumber_map).astype(int)
# Step 3 : Drop useless features

features_dropped = ['Name','Cabin','Ticket']

data_train = data_train.drop(features_dropped, axis = 1)

#data_train = data_train.drop(['PassengerId'], axis = 1)

data_test = data_test.drop(features_dropped, axis = 1)
data_train.info()
data_train.sample(3)
colormap = plt.cm.viridis

plt.figure(figsize=(13,13))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
sns.barplot(x='Fare',y='Survived', data=data_train)
sns.barplot(x='CabinNumber',y='Survived', hue='CabinFloor', data=data_train)
# Some useful parameters which will come in handy later on

ntrain = data_train.shape[0]

ntest = data_test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



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
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



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

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
# Create 5 objects that represent our 5 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = data_train['Survived'].ravel()

data_train_wpid = data_train.drop(['Survived','PassengerId'], axis=1)

x_train = data_train_wpid.values # Creates an array of the train data

data_test_wpid = data_test.drop(['PassengerId'], axis=1)

x_test = data_test_wpid.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
rf_feature = [ 0.11857778, 0.37834156, 0.06432355, 0.02969802, 0.02727931, 0.09343444, 0.02835798, 0.08336387, 0.05462566, 0.07344116, 0.0225561,  0.02600058]

et_feature = [ 0.12695597, 0.482216,  0.03816251, 0.02954247, 0.01974461, 0.06245438, 0.02788887, 0.04938868, 0.05796733, 0.03973576, 0.03143061, 0.0345128]

ada_feature = [ 0.018, 0.024, 0.186, 0.154, 0.124, 0.048, 0.024, 0.138, 0.01, 0.242, 0.01, 0.022]

gb_feature = [ 0.09349409, 0.05530585, 0.26488909, 0.05316861, 0.02960645, 0.17208911, 0.10793178, 0.08869151, 0.03721368, 0.06040429, 0.01820662, 0.01899892]
cols = data_train_wpid.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_feature,

     'Extra Trees feature importances': et_feature,

      'AdaBoost feature importances': ada_feature,

    'Gradient Boost feature importances': gb_feature

    })
feature_dataframe.head(15)
sns.barplot(x='features', y=feature_dataframe['AdaBoost feature importances'], data=feature_dataframe)
sns.barplot(x='features', y=feature_dataframe['Extra Trees feature importances'], data=feature_dataframe)
sns.barplot(x='features', y=feature_dataframe['Gradient Boost feature importances'], data=feature_dataframe)
sns.barplot(x='features', y=feature_dataframe['Random Forest feature importances'], data=feature_dataframe)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
from sklearn.metrics import accuracy_score

print(accuracy_score(base_predictions_train['AdaBoost'], y_train)*100)

print(accuracy_score(base_predictions_train['ExtraTrees'], y_train)*100)

print(accuracy_score(base_predictions_train['GradientBoost'], y_train)*100)

print(accuracy_score(base_predictions_train['RandomForest'], y_train)*100)
colormap = plt.cm.viridis

plt.figure(figsize=(4,4))

plt.title('Pearson Correlation of Models', y=1.05, size=15)

sns.heatmap(base_predictions_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
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
from sklearn.metrics import accuracy_score
print(accuracy_score(gbm.predict(x_train),y_train)*100)
PassengerId = data_test['PassengerId']

PassengerId.head()
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)