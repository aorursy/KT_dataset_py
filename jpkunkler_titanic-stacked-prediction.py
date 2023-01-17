import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb



# regular expressions

import re



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']
train.head()
full_data = [train, test]



# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Credit to Sina for her Titanic Best Working Classifier!



# create family size variable

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



# If family size == 1, a person is traveling by his/herself and for that is counted as 'Alone'

for dataset in full_data:

    dataset['IsAlone'] = 0 # default to 0, meaning not alone

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1 # Check for no companions

    

# If no information about embarkment, assume they joined in S

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')



# Use median imputation method for Fare column

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)



# Impute missing Age data

for dataset in full_data:

    avg_age = dataset['Age'].mean()

    std_age = dataset['Age'].std()

    null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(avg_age - std_age, avg_age + std_age, size=null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)



# define regular expression function to extract a person's title

def extract_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # if a title was found:

    if title_search:

        return title_search.group(1)

    return ""



# Create a new Column for Title

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(extract_title)

    

# since there are quite a lot of rare titles (e.g. 'Countess'), we'll group them as 'rare'

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') # Also replace french titles with common english abr.

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



# Now we have to transform everything categorical into numerical values by mapping

for dataset in full_data:

    

    # Transform gender into 0 for female and 1 for male

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    

    # Transform Age to a few categories

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age' ] = 1

    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age' ] = 2

    dataset.loc[ (dataset['Age'] > 38) & (dataset['Age'] <= 64), 'Age' ] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age' ] = 4

    

    # Map titles to values

    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_map)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Map City of Embarkment to value

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
# Now we'll drop any variables we don't need for our predictions 

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(), annot=True, cmap='viridis', linecolor='white', linewidths=0.1)
sns.pairplot(train, hue='Survived', diag_kind='kde', palette='Dark2')
# Import everything needed from SKLearn



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

#from sklearn.model_selection import KFold

from sklearn.cross_validation import KFold;
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

#kf = KFold(NFOLDS, random_state=SEED)

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

        return(self.clf.fit(x,y).feature_importances_)
# Out-Of-Fold Predictions

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
# Create objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Split data into train and test sets

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
rf_features = rf.feature_importances(x_train,y_train)

et_features = et.feature_importances(x_train, y_train)

ada_features = ada.feature_importances(x_train, y_train)

gb_features = gb.feature_importances(x_train,y_train)
# Create dataframe from above feature importances

cols = train.columns.values

feature_dataframe = pd.DataFrame({'features': cols,

              'Random Forest feature importances': rf_features,

              'Extra Trees feature importances': et_features,

              'AdaBoost feature importances': ada_features,

              'Gradient Boost feature importances': gb_features

             })
feature_dataframe.head()
plt.figure(figsize=(12,6))

plt.title('Random Forest feature importances', size=20)

sns.barplot(x='features', y='Random Forest feature importances', data=feature_dataframe)
plt.figure(figsize=(12,6))

plt.title('Gradient Boost feature importances', size=20)

sns.barplot(x='features', y='Gradient Boost feature importances', data=feature_dataframe)
plt.figure(figsize=(12,6))

plt.title('Extra Trees feature importances', size=20)

sns.barplot(x='features', y='Extra Trees feature importances', data=feature_dataframe)
plt.figure(figsize=(12,6))

plt.title('AdaBoost feature importances', size=20)

sns.barplot(x='features', y='AdaBoost feature importances', data=feature_dataframe)
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe
plt.figure(figsize=(12,6))

plt.title('Average Feature Importance', size=20)

sns.barplot(x='features', y='mean', data=feature_dataframe)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
sns.heatmap(base_predictions_train.astype(float).corr(), cmap='viridis')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
# Create and fit the Gradient Boosting Model

gbm = xgb.XGBClassifier(

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1)



gbm.fit(x_train, y_train)



# Predict survival on test data

predictions = gbm.predict(x_test)
# Generate Submission File 

Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

Submission.to_csv("Submission.csv", index=False)