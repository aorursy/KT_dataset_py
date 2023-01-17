import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold
#Read in the dataset, and giving back their headers

data = pd.read_csv('../input/pima-indians-diabetes.csv', header = None, names = 

                   ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',

                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class'])
data.head()
data.describe()
data.info()
colormap = plt.cm.inferno

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
#Separate the predictors and response variables

y = data['Class'].ravel()

X = data.drop(['Class'], axis=1)
#Defining some variables

SEED = 2019 # for reproducibility

NFOLDS = 3

kfold = KFold(n_splits = NFOLDS, random_state = SEED)
#Separate training and testing dataset

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = SEED)
ntrain = x_train.shape[0]  #Number of rows per training column (recall: train.shape = (3, 4), then shape[0] is 3)

ntest = x_test.shape[0]  #testing column

x_train = x_train.values
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

    oof_train = np.zeros((ntrain,))  #Establish an array of 0s that has the same length of the training data

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))  #Establish an array consist of 5 rows and number of test columns



    for i, (train_index, test_index) in enumerate(kfold.split(x_train)):  #Elocate the splited data into their dataset

        x_tr = x_train[train_index]   #train index of train dataset

        y_tr = y_train[train_index]   #train index of test dataset

        x_te = x_train[test_index]  #test index of train dataset



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 300,

     'warm_start': True, 

     'max_features': 0.7,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':300,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.1

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     'max_features': 0.7,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
#Checking feature importance

rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
data = [

    go.Heatmap(

        z = base_predictions_train.astype(float).corr().values ,

        x = base_predictions_train.columns.values,

        y = base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
import xgboost as xgb
gbm = xgb.XGBClassifier(learning_rate = 0.02,

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



accuracy_score(y_test, predictions)