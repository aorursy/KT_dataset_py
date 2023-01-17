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

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;
print( "\nReading data from disk ...")

properties = pd.read_csv(r"../input/properties_2016.csv")

train_df = pd.read_csv("../input/train_2016_v2.csv")

test_df = pd.read_csv("../input/sample_submission.csv")

test_df = test_df.rename(columns={'ParcelId': 'parcelid'})
train = train_df.merge(properties, how = 'left', on = 'parcelid')

test = test_df.merge(properties, on='parcelid', how='left')
from sklearn.preprocessing import LabelEncoder  



lbl = LabelEncoder()



for c in train.columns:

    train[c]=train[c].fillna(0)

    if train[c].dtype == 'object':

        lbl.fit(list(train[c].values))

        train[c] = lbl.transform(list(train[c].values))



for c in test.columns:

    test[c]=test[c].fillna(0)

    if test[c].dtype == 'object':

        lbl.fit(list(test[c].values))

        test[c] = lbl.transform(list(test[c].values))     
from sklearn import model_selection, preprocessing

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")



train_y = train.logerror.values

train_X = train.drop(["parcelid", "transactiondate", "logerror"], axis=1)

xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
featureImportance = model.get_fscore()

features = pd.DataFrame()

features['features'] = featureImportance.keys()

features['importance'] = featureImportance.values()

features.sort_values(by=['importance'],ascending=False,inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

plt.xticks(rotation=90)

sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h", color = "#34495e")
topFeatures = features["features"].tolist()[:20]

corrMatt = train[topFeatures].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

colormap = plt.cm.viridis

plt.figure(figsize=(12,20))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(corrMatt,linewidths=0.6,vmax=1.0, mask = mask, square = True, linecolor='white', annot=True)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(train.logerror.values, bins=500, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
train=train[ train.logerror > -0.40 ]

train=train[ train.logerror < 0.419 ]



plt.figure(figsize=(12,8))

sns.distplot(train.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
test.head(2)
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



# Class to extend the Sklearn Regressor

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

    

# Class to extend XGboost classifer
print(train.shape)

print(test.shape)
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
# Put in our parameters for said regressors

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 50,

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

    'n_estimators':50,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 50,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 50,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}

#Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)

#svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
feature_names = list(train.columns)

print(np.setdiff1d(train.columns, test.columns))
do_not_include = ['parcelid', 'logerror', 'transactiondate', 'hashottuborspa',

 'propertycountylandusecode',

 'propertyzoningdesc',

 'fireplaceflag',

 'taxdelinquencyflag']



feature_names = [f for f in train.columns if f not in do_not_include]



print("We have %i features."% len(feature_names))

train[feature_names].count()
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['logerror'].ravel()

#train = train.drop(['logerror', 'parcelid', 'transactiondate'], axis=1)

train = train[feature_names]

test = test[feature_names]
print(train.shape)

print(test.shape)
x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

#svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector 



print("Training is complete")
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
data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Portland',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
X = x_train

y = y_train

y_mean = np.mean(y_train)
from sklearn.model_selection import train_test_split



Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=2000)



dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)



watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Try different parameters! My favorite is random search :)

xgb_params = {

    'eta': 0.025,

    'max_depth': 7,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 0.8,   

    'alpha': 0.4, 

    'base_score': y_mean,

    'silent': 1

}

model_xgb = xgb.train(xgb_params, dtrain, 2000, watchlist, early_stopping_rounds=300,

                  maximize=False, verbose_eval=15)
dtest = xgb.DMatrix(x_test)

predicted_test_xgb = model_xgb.predict(dtest)
sub = pd.read_csv('../input/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:

    sub[c] = predicted_test_xgb



print('Writing csv ...')

sub.to_csv('xgb_stacked.csv', index=False, float_format='%.4f')