# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

# Load in our libraries

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



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LinearRegression



from keras.models import Sequential

from keras.layers.core import Dense, Dropout

from keras.optimizers import Adam, SGD
# Load in the train and test datasets

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
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
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict_proba(x)[:,1]

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

    

# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):

    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    kf = kfold.split(x_train, y_train)

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        y_te = y_train[test_index]



        clf.train(x_tr, y_tr)

        clf.clf = CalibratedClassifierCV(clf.clf, cv='prefit')

        clf.clf.fit(x_te, y_te)

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train, oof_test
# Put in our parameters for said classifiers

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
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 



print("Training is complete")
def get_nn_model():

    nn_model = Sequential()

    nn_model.add(Dense(32,activation='relu',input_shape=(11,)))

    nn_model.add(Dropout(0.3))

    nn_model.add(Dense(64,activation='relu'))

    nn_model.add(Dropout(0.3))

    nn_model.add(Dense(64,activation='relu'))

    nn_model.add(Dropout(0.3))

    nn_model.add(Dense(1, activation='sigmoid'))



    Loss = 'binary_crossentropy'

    nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])

    return nn_model
cv_score = []

dn_cv_test = np.zeros(len(PassengerId))

dn_cv_train = np.zeros((ntrain,))

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

kf = kfold.split(x_train, y_train)

for i, (train_fold, validate) in enumerate(kf):



    X_train = x_train[train_fold]

    label_train = y_train[train_fold]

    X_validate = x_train[validate]

    label_validate = y_train[validate]

    nn_model = get_nn_model()

    history = nn_model.fit(X_train,label_train, batch_size=64, epochs=30, 

                           validation_data=(X_validate, label_validate), verbose=0)

    cv_score.append(nn_model.evaluate(X_validate, label_validate)[1])

    dn_cv_train[validate] = nn_model.predict(X_validate).T[0]

    dn_cv_test += nn_model.predict(x_test).T[0]

    

print('cv scores:', cv_score)

print('cv score:', np.mean(cv_score))

dn_cv_test /= NFOLDS
prediction = dn_cv_test

prediction = prediction > 0.5

prediction = prediction.astype(np.int)

nn_model_submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': prediction })

nn_model_submission.to_csv("dn_submission.csv", index=False)

nn_model_submission
print(pd.DataFrame(et_oof_train).describe())

print(pd.DataFrame(ada_oof_train).describe().round(6))

print(pd.DataFrame(dn_cv_train).describe())
base_predictions_train = pd.DataFrame( {

     'ExtraTrees': et_oof_train,

     'AdaBoost': ada_oof_train,

      'DenseNetwork': dn_cv_train,

    })

base_predictions_train.head()
x_train = np.concatenate((et_oof_train.reshape(-1, 1), ada_oof_train.reshape(-1, 1), dn_cv_train.reshape(-1, 1)), axis=1)

x_test = np.concatenate((et_oof_test.reshape(-1, 1), ada_oof_test.reshape(-1, 1), dn_cv_test.reshape(-1, 1)), axis=1)
reg = LinearRegression()

reg.fit(x_train, y_train)

reg_predict = reg.predict(x_test)

reg_predict = reg_predict > 0.5

reg_predict = reg_predict.astype(np.int)

reg_model_submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': reg_predict })

reg_model_submission.to_csv("reg_model_submission.csv", index=False)

reg_model_submission