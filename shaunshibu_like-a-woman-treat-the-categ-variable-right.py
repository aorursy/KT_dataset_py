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

sample_submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

test = pd.read_csv("../input/cat-in-the-dat/test.csv")

train = pd.read_csv("../input/cat-in-the-dat/train.csv")
print(train.info())
train
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

carrier_count = train['ord_1'].value_counts()

sns.set(style="darkgrid")

sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)

plt.title('Frequency Distribution of Carriers')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('nom_7', fontsize=12)

plt.show()

train.dtypes
scale_mapper = {'Novice':1, 

                'Contributor':2,

                'Master':3,

                'Grandmaster':4,

                'Expert':5}
scale_mapper2 = {'Freezing':1, 

                'Cold':2,

                'Warm':3,

                'Hot':4,

                'Boiling Hot':5,

                'Lava Hot':6}
scale_mapper3 = {'a':1, 

                'b':2,

                'c':3,

                'd':4,

                'e':5,

                'f':6, 

                'g':7,

                'h':8,

                'i':9,

                'j':10,

                'k':11, 

                'l':12,

                'm':13,

                'n':14,

                'o':15,

                }
scale_mapper4 = {'A':1, 

                'B':2,

                'C':3,

                'D':4,

                'E':5,

                'F':6, 

                'G':7,

                'H':8,

                'I':9,

                'J':10,

                'K':11, 

                'L':12,

                'M':13,

                'N':14,

                'O':15,

                'P':16, 

                'Q':17,

                'R':18,

                'S':19,

                'T':20,

                'U':21, 

                'V':22,

                'W':23,

                'X':24,

                'Y':25,

                'Z':26}
train['experience'] = train['ord_1'].replace(scale_mapper)
test['experience'] = test['ord_1'].replace(scale_mapper)
train['temperature'] = train['ord_2'].replace(scale_mapper2)
test['temperature'] = test['ord_2'].replace(scale_mapper2)
train['small'] = train['ord_3'].replace(scale_mapper3)
test['small'] = test['ord_3'].replace(scale_mapper3)
train['capital'] = train['ord_4'].replace(scale_mapper4)
test['capital'] = test['ord_4'].replace(scale_mapper4)
binaryconverter = {'T':1,

                   'F':0}
binaryconverter2 = {'Y':1,

                   'N':0}
train['truefalse'] = train['bin_3'].replace(binaryconverter)

test['truefalse'] = test['bin_3'].replace(binaryconverter)

train['yesno'] = train['bin_4'].replace(binaryconverter2)
test['yesno'] = test['bin_4'].replace(binaryconverter2)
train.rename(columns={'bin_0':'a','bin_1':'b'},inplace=True)
test.rename(columns={'bin_0':'a','bin_1':'b'},inplace=True)
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train['color'] = lb_make.fit_transform(train['nom_0'])



train.head()

import numpy as np
#Function defined to apply mean encoding technique to nominal features and k-fold to apply regularization andf prevent overfitting

from sklearn import base

from sklearn.model_selection import KFold

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):



    def __init__(self, colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):



        self.colnames = colnames

        self.targetName = targetName

        self.n_fold = n_fold

        self.verbosity = verbosity

        self.discardOriginal_col = discardOriginal_col



    def fit(self, X, y=None):

        return self





    def transform(self,X):



        assert(type(self.targetName) == str)

        assert(type(self.colnames) == str)

        assert(self.colnames in X.columns)

        assert(self.targetName in X.columns)



        mean_of_target = X[self.targetName].mean()

        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=2019)







        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'

        X[col_mean_name] = np.nan



        for tr_ind, val_ind in kf.split(X):

            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

#             print(tr_ind,val_ind)

            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())



        X[col_mean_name].fillna(mean_of_target, inplace = True)



        if self.verbosity:



            encoded_feature = X[col_mean_name].values

            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,

                                                                                      self.targetName,

                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))

        if self.discardOriginal_col:

            X = X.drop(self.targetName, axis=1)

            



        return X
from sklearn import base

from sklearn.model_selection import KFold



targetc = KFoldTargetEncoderTrain('nom_1','target',n_fold=5)

train = targetc.fit_transform(train)
#train.rename(columns={'nom_1_Kfold_Target_Enc':'shape'},inplace=True)
train.columns
#Mean Encoding categorical variable with K-fold regularization to handle nominal variables

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

    

    def __init__(self,train,colNames,encodedName):

        

        self.train = train

        self.colNames = colNames

        self.encodedName = encodedName

        

        

    def fit(self, X, y=None):

        return self



    def transform(self,X):





        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 

        

        dd = {}

        for index, row in mean.iterrows():

            dd[row[self.colNames]] = row[self.encodedName]



        

        X[self.encodedName] = X[self.colNames]

        X = X.replace({self.encodedName: dd})



        return X
#Shapes ---> test

test_targetc = KFoldTargetEncoderTest(train,'nom_1','nom_1_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_2','target',n_fold=5)

train = targetc.fit_transform(train)
#Animals ---> test

test_targetc = KFoldTargetEncoderTest(train,'nom_2','nom_2_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_3','target',n_fold=5)

train = targetc.fit_transform(train)
#Countries

test_targetc = KFoldTargetEncoderTest(train,'nom_3','nom_3_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_4','target',n_fold=5)

train = targetc.fit_transform(train)
#Instruments

test_targetc = KFoldTargetEncoderTest(train,'nom_4','nom_4_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_5','target',n_fold=5)

train = targetc.fit_transform(train)
#

test_targetc = KFoldTargetEncoderTest(train,'nom_5','nom_5_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_6','target',n_fold=5)

train = targetc.fit_transform(train)
#

test_targetc = KFoldTargetEncoderTest(train,'nom_6','nom_6_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_7','target',n_fold=5)

train = targetc.fit_transform(train)
#

test_targetc = KFoldTargetEncoderTest(train,'nom_7','nom_7_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_8','target',n_fold=5)

train = targetc.fit_transform(train)
#

test_targetc = KFoldTargetEncoderTest(train,'nom_8','nom_8_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('nom_9','target',n_fold=5)

train = targetc.fit_transform(train)
#

test_targetc = KFoldTargetEncoderTest(train,'nom_9','nom_9_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
targetc = KFoldTargetEncoderTrain('ord_5','target',n_fold=5)

train = targetc.fit_transform(train)
#

test_targetc = KFoldTargetEncoderTest(train,'ord_5','ord_5_Kfold_Target_Enc')

test = test_targetc.fit_transform(test)
train.dtypes
#Shapes Animals Countries Instruments

train.rename(columns={'nom_1_Kfold_Target_Enc':'shape','nom_2_Kfold_Target_Enc':'animal','nom_3_Kfold_Target_Enc':'country','nom_4_Kfold_Target_Enc':'instrument','nom_5_Kfold_Target_Enc':'p','nom_6_Kfold_Target_Enc':'q','nom_7_Kfold_Target_Enc':'r','nom_8_Kfold_Target_Enc':'s','nom_9_Kfold_Target_Enc':'t','ord_5_Kfold_Target_Enc':'u'},inplace=True)
#Shapes Animals Countries Instruments

test.rename(columns={'nom_1_Kfold_Target_Enc':'shape','nom_2_Kfold_Target_Enc':'animal','nom_3_Kfold_Target_Enc':'country','nom_4_Kfold_Target_Enc':'instrument','nom_5_Kfold_Target_Enc':'p','nom_6_Kfold_Target_Enc':'q','nom_7_Kfold_Target_Enc':'r','nom_8_Kfold_Target_Enc':'s','nom_9_Kfold_Target_Enc':'t','ord_5_Kfold_Target_Enc':'u'},inplace=True)
final_train = train[['a','b','truefalse','yesno','experience','temperature','small','capital','color','shape','animal','country','instrument','p','q','r','s','t','u','day','month','target']]
final_test = train[['a','b','truefalse','yesno','experience','temperature','small','capital','color','shape','animal','country','instrument','p','q','r','s','t','u','day','month']]
final_train
#Handling imbalanced features using SMOTE oversampling technique

import numpy as np

from sklearn.model_selection import train_test_split

X = final_train.iloc[:, 0:20]

y = final_train.iloc[:, 21]

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns )

os_data_y= pd.DataFrame(data=os_data_y,columns=['target'])

# we can Check the numbers of our data

print("length of oversampled data is ",len(os_data_X))

print("Number of 0 in oversampled data",len(os_data_y[os_data_y['target']==0]))

print("Number of subscription",len(os_data_y[os_data_y['target']==1]))

print("Proportion of target 0 in oversampled data is ",len(os_data_y[os_data_y['target']==0])/len(os_data_X))

print("Proportion of target 1 in oversampled data is ",len(os_data_y[os_data_y['target']==1])/len(os_data_X))
#Applying XGBoost Model

#Scope for improvement --> hyperparameter grid tuning

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
final_train = pd.concat([os_data_X,os_data_y],axis=1)
train, test = train_test_split(final_train, test_size=0.4)
train

X_train = train.iloc[:,0:19]

y_train = train.iloc[:,20]
test
X_test = test.iloc[:,0:19]

y_test = test.iloc[:,20]
X_valid = final_test.iloc[:,0:20]

model = XGBClassifier(max_depth=8, objective='reg:logistic',eta=0.3,subsample=0.8,colsample_bytree =0.9,colsample_bylevel=1,min_child_weight=10,num_boost_round=200, 

                  early_stopping_rounds=30, maximize=False, 

                  verbose_eval=10)

model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
final_train
Xtrain = final_train.iloc[:,0:19]

Ytrain = final_train.iloc[:,20]
# stratified k-fold cross validation evaluation of xgboost model ------> read more about this!!!

from numpy import loadtxt

import xgboost

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



# CV model

model2 = xgboost.XGBClassifier(max_depth=8, objective='reg:logistic',eta=0.3,subsample=0.8,colsample_bytree =0.9,colsample_bylevel=1,min_child_weight=10)

kfold = StratifiedKFold(n_splits=10, random_state=7)

results = cross_val_score(model2, Xtrain, Ytrain, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(model.feature_importances_)

from xgboost import plot_importance 

import matplotlib.pyplot as plt

fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

plot_importance(model,ax = axes,height = 0.5)



plt.show()
import lightgbm as lgb

import pandas as pd

from sklearn.metrics import mean_squared_error

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



# specify your configurations as a dict

params = {'boosting_type':'gbdt',

        'objective': 'regression',

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 5000 ,

        'bagging_freq': 20,

        'colsample_bytree': 0.6,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'zero_as_missing': True,

        'seed':0,        

    }







print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=1000,

                valid_sets=lgb_eval,

                verbose_eval=50,

                early_stopping_rounds=5)



y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
import lightgbm as lgbm

fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgbm.plot_importance(gbm,ax = axes,height = 0.5)

plt.show();plt.close()
#Applying regression models

#Logistic Regression

final_train
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score

lr=LogisticRegression(C=0.125, solver="lbfgs", max_iter=500)  



lr.fit(X_train, y_train)

y_pred_lr=lr.predict_proba(X_test)

roc_auc_score(y_test.values, y_pred_lr[:,1])
import sklearn.metrics as metrics

y_pred_proba = lr.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()

import numpy as np

from sklearn import linear_model, datasets

from sklearn.model_selection import GridSearchCV

# Create regularization penalty space



penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)

logistic = linear_model.LogisticRegression()
# Create grid search using 5-fold cross validation

clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(X_train, y_train)
y_pred_lr2=clf.predict_proba(X_test)
roc_auc_score(y_test.values, y_pred_lr2[:,1])
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])