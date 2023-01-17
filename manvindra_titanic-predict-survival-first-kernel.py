#This cell contains basic code from Kaggle and following cells follows outlines and code from Manav Sehgal notebook(Titanic Data Science Solutions)

# Also took learning with code from from https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

# Learning Python using the above notebooks.

#This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# visualization libs

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rc('font', family='sans-serif') 

plt.rc('font', serif='Helvetica Neue') 

plt.rc('text', usetex='false') 

plt.rcParams.update({'font.size': 10})
#Import ML Classfication libs

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.model_selection  import KFold;

import xgboost as xgb
#Acquire data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df] #will be helpful in finding all distinct titles.
print(train_df.columns.values)

train_df.head()
#Data Type of features

train_df.info()

print("------------------")

test_df.info()
#Distribution of numerical features

train_df.describe()
#Distribution of Categorical data

train_df.describe(include=['O'])

#Cabin has lot of null, drop it

train_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)

#Drop ticket number also

train_df.drop("Ticket",axis=1,inplace=True)

test_df.drop("Ticket",axis=1,inplace=True)
train_df.hist(bins=10,figsize=(10, 10),grid=True);
# Embarked =S, Pclass=3, and No SibSp has large set of passenger who didn't survived

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0],ax=axis1)

sns.countplot(x='Survived', hue="Pclass", data=train_df, order=[1,0],ax=axis2)

sns.countplot(x='Survived', hue="SibSp", data=train_df, order=[1,0],ax=axis3)
# Remove all NULLS in the Embarked column

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)

# Most passenger above 60 yr of age didn't survive.

# Childern survival rate is higher.

#Consider for Model training
#Age important variable, Fill Null value with random assigment between +-1SD from mean

average_age_train   = train_df["Age"].mean()

std_age_train       = train_df["Age"].std()

count_nan_age_train = train_df["Age"].isnull().sum()

average_age_test   = test_df["Age"].mean()

std_age_test       = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()





random_age1=np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)

random_age2=np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

train_df["Age"][np.isnan(train_df["Age"])] = random_age1

test_df["Age"][np.isnan(test_df["Age"])] = random_age2

train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)

train_df['CategoricalAge'] = pd.cut(train_df['Age'], 5)
g = sns.FacetGrid(train_df, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age");

#Female has higher survival rate
g = sns.FacetGrid(train_df, hue="Survived", col="Pclass", margin_titles=True)

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

#High Class and Fare have better survival rate. Create band of fare (Think Decision Tree split)

#Fill Null

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
#Fare null values imputation

for dataset in combine:

    dataset['Fare'] = dataset['Fare'].fillna(train_df['Fare'].median())
#Fare categories 

train_df['CategoricalFare'] = pd.qcut(train_df['Fare'], 4)
# from https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

#Title from names

# Define function to extract titles from passenger names

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in combine:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')  

for dataset in combine:

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

    
train_df.head()
# Feature selection

drop_elements = ['PassengerId', 'Name']

train_set = train_df.drop(train_df.columns[3], axis = 1)

train_set = train_set.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test_set  = test_df.drop(drop_elements, axis = 1)
test_set.head()
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_set.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#training ,test data

train = train_set.drop(["Survived","PassengerId"] , axis=1)

x_train = train.values

y_train = train_set["Survived"].ravel()

x_test = test_set.values

#x_train.shape, y_train.shape, x_test.shape
# Some useful parameters which will come in handy later on

ntrain = train_set.shape[0]

ntest = test_set.shape[0]

print(ntrain,ntest)

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits= NFOLDS, random_state=SEED)



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

        return(self.clf.fit(x,y).feature_importances_)
#Out of Fold Prediction

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))

    i=0;

    for train_index, test_index in kf.split(x_train):

        x_tr, x_te = x_train[train_index], x_train[test_index]

        y_tr = y_train[train_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)

        i=i+1

    oof_test[:] = oof_test_skf.mean(axis=0)

    

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#Setting params for classifiers

# Random Forest params

rf_params = {

    'n_jobs' : -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt'

}

# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 86,

    'min_samples_leaf': 2

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

    'min_samples_leaf': 2

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
#Create object of each classifier

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
#fit 

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("_____ Complete")
#Feature Importance

rf_feature=rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest Feat': rf_feature,

     'Extra Trees Feat': et_feature,

      'AdaBoost Feat': ada_feature,

    'GB Feat': gb_feature

    })
feature_dataframe.head()
fig, axs = plt.subplots(figsize=(20,10), ncols=2, nrows=2)



g=sns.stripplot(y=feature_dataframe['Random Forest Feat'].values,

              x=feature_dataframe['features'].values, data=feature_dataframe

               ,size=20,ax=axs[0][0]);

g.axes.set_title('Randrom Forest feature importance', fontsize=20,color="r")



g=sns.stripplot(y=feature_dataframe['Extra Trees Feat'].values,

              x=feature_dataframe['features'].values, data=feature_dataframe

               ,size=20,ax=axs[0][1]);

g.axes.set_title('Extra Trees feature importance', fontsize=20,color="r")



g=sns.stripplot(y=feature_dataframe['AdaBoost Feat'].values,

              x=feature_dataframe['features'].values, data=feature_dataframe

               ,size=20,ax=axs[1][0]);

g.axes.set_title('Adaboost feature importance', fontsize=20,color="r")



g=sns.stripplot(y=feature_dataframe['GB Feat'].values,

              x=feature_dataframe['features'].values, data=feature_dataframe

               ,size=20,ax=axs[1][1]);

g.axes.set_title('GB feature importance', fontsize=20,color="r")

# Create the new column containing the average of values



feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(3)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
sns.heatmap(base_predictions_train.astype(float).corr().values, 

        xticklabels=base_predictions_train.columns.values,

        yticklabels=base_predictions_train.columns.values)
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
x_train.shape
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

 n_jobs= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
# Generate Submission File 

Submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions })

Submission.to_csv("Submission.csv", index=False)