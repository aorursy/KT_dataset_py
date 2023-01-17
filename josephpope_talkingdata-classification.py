import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import os
#print(os.listdir("../input"))

import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
auc, recall_score, roc_curve, roc_auc_score)
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

%matplotlib inline

# Any results you write to the current directory are saved as output.
# seaborn settings 

sns.set(rc={'figure.figsize':(10,4)});
plt.figure(figsize=(10,4));
print(sns.__version__)
train = pd.read_csv('../input/train.csv', nrows=18790469)
test = pd.read_csv('../input/test.csv', nrows=18790469)
test.head()
train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
train.nunique()
train['click_time'] = train['click_time'].astype('datetime64[ns]')
test['click_time'] = test['click_time'].astype('datetime64[ns]')
train.dtypes
train.head(1)
train.columns
user_agent = ['ip','app','device','os','channel']
# concatenate all computer related features into one string for data exploration purposes. 
# we should be reasonably certain that each user agent is one unique individual. 
# click time and attributed time may help shed further light here. 

train['user_agent'] = train[user_agent].apply(lambda x: '.'.join(x.astype(str)), axis=1)
train.dtypes
train['day'] = train['click_time'].dt.day.astype('uint8')
train['hour'] = train['click_time'].dt.hour.astype('uint8')
train['minute'] = train['click_time'].dt.minute.astype('uint8')
train['second'] = train['click_time'].dt.second.astype('uint8')
test['day'] = test['click_time'].dt.day.astype('uint8')
test['hour'] = test['click_time'].dt.hour.astype('uint8')
test['minute'] = test['click_time'].dt.minute.astype('uint8')
test['second'] = test['click_time'].dt.second.astype('uint8')
train.head(1)
# prepare X and y for train, test splits and CV 

y = train['is_attributed']
X = train.drop(['ip', 'is_attributed','click_time',
                'attributed_time'], axis=1).select_dtypes(include=[np.number])
test.columns
test = test.drop(['click_time'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state=44)
# Split into X and y

# Create a model
# Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
clf_xgBoost = xgb.XGBClassifier(
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.7,
    colsample_bylevel = 0.7,
    scale_pos_weight = 2, # default 9 
    min_child_weight = 0,
    reg_alpha = 4,
    n_jobs = 4, 
    objective = 'binary:logistic'
)
# Fit the models
model = clf_xgBoost.fit(X_train, y_train)
# default ROC score is .9455
def summary_stats(model, X_train=X_train, y_train=y_train, 
                  X_test=X_test, y_test=y_test):
    '''
    input model (linear regression, random forest), X_train, y_train, X_test and y_test
    '''
    model.fit(X_train, y_train)
    
    try: 
        prob_y = model.predict_proba(X_test)
        prob_y = [p[1] for p in prob_y]
    except: 
        pass
    
    p = model.predict(X_test)
    
    print("ROC Score: {:.04f}\nRecall Score: {:.04f}\n\
Accuracy Score: {:.04f}\nMisclassification Rate: {:04f}\n\
Precision Score: {:.04f}\nF1 Score: {:.04f}\n"
          .format(roc_auc_score(y_test, prob_y),
                  recall_score(y_test,p),
                  accuracy_score(y_test,p),
                  1-accuracy_score(y_test,p),
                  precision_score(y_test,p),
                  f1_score(y_test,p)))
    
    print(confusion_matrix(y_test, p))
    print("\n")
    print(classification_report(y_test, p))
summary_stats(clf_xgBoost)
submission_cols = ['click_id','is_attributed']
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]
params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'scale_pos_weight': 150,
          'silent': True}
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 270, watchlist, maximize=True, verbose_eval=10)
test.drop(['click_id','ip'], axis=1, inplace=True)
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub.csv',index=False)