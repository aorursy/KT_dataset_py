# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

submission=pd.read_csv('../input/sample_submission.csv')
#lets fit a randomforest 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from random import *

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report
train_id=train['id']

train=train.drop('id',1)

target=train['target']

train=train.drop('target',1)

test_id=test['id']

test=test.drop('id',1)
sample_weight1= np.array([10 if i == 1 else 1 for i in target])

sample_weight3 = np.array([25 if i == 1 else 1 for i in target])

sample_weight2 = np.array([20 if i == 1 else 1 for i in target])
samples=[sample_weight1,sample_weight2,sample_weight3]
import random



def some(df):

    return df.iloc[np.random.choice(df.index,28000)]
X_train, X_test, y_train, y_test = train_test_split(train,target, test_size=0.20, random_state=randint(1, 100))

clf = RandomForestClassifier(n_estimators=50)

model=clf.fit(X_train,y_train)

y_pred=model.predict(X_test)

    # evaluate all types of errors 

target_names = ['class 0', 'class 1']

print(classification_report(y_test, y_pred, target_names=target_names))

    
import xgboost as xgb

#from xgboost.sklearn import XGBClassifier

#from sklearn import cross_validation, metrics 

params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'binary:logistic',

    

    'eval_metric': 'error',

    'silent': 0

}

dtrain = xgb.DMatrix(X_train,y_train)

model=xgb.train(params,dtrain)
set(y_pred)
y_pred=model.predict(xgb.DMatrix(X_test))

y_pred=list(y_pred)

for i in range(len(y_pred)):

    if y_pred[i]>0.45:

        y_pred[i]=1

    else:

        y_pred[i]=0

    # evaluate all types of errors 

target_names = ['class 0', 'class 1']

print(classification_report(y_test, y_pred, target_names=target_names))

    
fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
train_data=pd.read_csv('../input/train.csv')
train_data[train_data['target']==1]['ps_ind_05_cat'].value_counts().plot(kind='bar')
train_data[train_data['target']==0]['ps_ind_05_cat'].value_counts().plot(kind='bar')
train_data[train_data['target']==1]['ps_ind_04_cat'].value_counts().plot(kind='bar')
train_data[train_data['target']==0]['ps_ind_04_cat'].value_counts().plot(kind='bar')
submission['id']=test_id

submission['target']=model.predict(xgb.DMatrix(test))

submission['target']=submission['target'].apply(lambda x: 0 if x<=0.45 else 1)

submission.to_csv('itsonlyxgb.csv',index=0)
train_data=pd.read_csv('../input/train.csv')

proper_cross_validation_data=some(train_data)

cross_validation_target=proper_cross_validation_data['target']

proper_cross_validation_data=proper_cross_validation_data.drop(['id','target'],1)

train_data=train_data.loc[[c for c in train_data.index if c not in proper_cross_validation_data.index]]



train_data_positive=train_data[train_data['target']==1]

train_data_negative=train_data[train_data['target']==0]

train_data_negative=train_data_negative[:20700]

train_data=pd.concat([train_data_positive,train_data_negative])

train_target=train_data['target']

train_data=train_data.drop(['id','target'],1)
#X_train, X_test, y_train, y_test = train_test_split(train_data,train_target, test_size=0.20, random_state=randint(1, 100))

clf = RandomForestClassifier(n_estimators=50)

model=clf.fit(train_data,train_target)

y_pred=model.predict(proper_cross_validation_data)

    # evaluate all types of errors 

target_names = ['class 0', 'class 1']

print(classification_report(cross_validation_target, y_pred, target_names=target_names))

    
# feature importances

import matplotlib.pyplot as plt

(clf.feature_importances_)
importance = clf.feature_importances_

importance = pd.DataFrame(importance, index=train_data.columns, 

                          columns=["Importance"])



importance["Std"] = np.std([tree.feature_importances_

                            for tree in clf.estimators_], axis=0)



x = range(importance.shape[0])

y = importance.iloc[:, 0]

yerr = importance.iloc[:, 1]



plt.bar(x, y, yerr=yerr, align="center")



plt.show()
list(zip(train_data, clf.feature_importances_))
l1=pd.DataFrame(l1)
features= l1.sort_values(1, ascending=False)
features.head()[0]
train_data_xgb=train_data.drop(['ps_ind_05_cat','ps_reg_01'],1)
import xgboost as xgb

#from xgboost.sklearn import XGBClassifier

#from sklearn import cross_validation, metrics 

params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'binary:logistic',

    

    'eval_metric': 'error',

    'silent': 0

}

dtrain = xgb.DMatrix(train_data,train_target)

model=xgb.train(params,dtrain)
proper_cross_validation_data_xgb=proper_cross_validation_data.drop(['ps_ind_05_cat','ps_reg_01'],1)
y_pred=model.predict(xgb.DMatrix(proper_cross_validation_data))

y_pred=list(y_pred)

for i in range(len(y_pred)):

    if y_pred[i]>0.45:

        y_pred[i]=1

    else:

        y_pred[i]=0

    # evaluate all types of errors 

target_names = ['class 0', 'class 1']

print(classification_report(cross_validation_target, y_pred, target_names=target_names))

    
#bst = xgb.cv(params, dtrain, num_boost_round=100, early_stopping_rounds= 40,nfold=5 ,verbose_eval=10)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
xgb_top_features=['ps_car_13','ps_reg_03','ps_ind_05_cat','ps_reg_01','ps_car_14']
features.head()[0]
#plotting these 10 features 
model
submission['id']=test_id

submission['target']=model.predict(xgb.DMatrix(test))

submission.to_csv('itsxgb.csv.csv',index=0)
train=train.drop(['id','target'],1)
dtrain = xgb.DMatrix(train_data,train_target)

model=xgb.train(params,dtrain)



y_pred=model.predict(xgb.DMatrix(proper_cross_validation_data))

y_pred=list(y_pred)

for i in range(len(y_pred)):

    if y_pred[i]>0.50:

        y_pred[i]=1

    else:

        y_pred[i]=0

    # evaluate all types of errors 

target_names = ['class 0', 'class 1']

print(classification_report(cross_validation_target, y_pred, target_names=target_names))

    
test=test.drop('id',1)
X_test=xgb.DMatrix(test)
submission['target']=model.predict(X_test)
import matplotlib.pyplot as plt
submission.shape
plt.plot(submission['target'])

plt.show()
submission[submission['target']>=0.4]
submission['target']=submission['target'].apply(lambda x: (1  if x>=0.4 else 0))
submission['target']
plt.plot(model.get_fscore())

plt.show()
model.get_fscore()
len(model.get_fscore())
model.get_score()
dir(model)
plt.plot(model.get_split_value_histogram)

#submission.to_csv('xgb.csv',index=0)#
#submission.to_csv('xgb2.csv',index=0)
list(submission)
#precision recall dielema 