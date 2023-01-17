# this version gave a 0.96xxx overall score
# special handling of data was required
# 03/01/2016

import os
import sys
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

sys.path.append("C:/Continuum/Anaconda3/Lib/site-packages/xgboost-master/python-package/")
sys.path.append("C:/Continuum/Anaconda3/Lib/site-packages/xgboost-master/python-package/xgboost/")
sys.path.append("C:/Continuum/Anaconda3/Lib/site-packages/xgboost-master/windows/x64/Release/")

import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

from time import time

pd.set_option('display.max_rows',9999)

seed = 999
t0 = time()
#train = pd.read_csv("../input/train.csv")
#test = pd.read_csv("../input/test.csv")
d1 = pd.read_csv("../input/train.csv", parse_dates=[1])

y = d1['QuoteConversion_Flag']
del d1['QuoteConversion_Flag']

d1['year'] = d1['Original_Quote_Date'].dt.year
d1['month'] = d1['Original_Quote_Date'].dt.month
d1['wkday'] = d1['Original_Quote_Date'].dt.dayofweek

del d1['Original_Quote_Date']

tot_features = d1.columns

missing_data_list = []

for v in tot_features:
    n1 = d1[v].isnull().sum()
    if (n1 > 0):
#       print(v, n1)
        missing_data_list.append(v)

#print(missing_data_list)

unique_keys = []

for v in missing_data_list:
    w = d1[v].unique()
    unique_keys.append(w)
    
n1 = len(missing_data_list)

d1['PropertyField29'].fillna(-1, inplace=True)

d1['PersonalField7'].fillna('-1', inplace=True)
d1['PersonalField84'].fillna(0, inplace=True)

d1['PropertyField3'].fillna('-1', inplace=True)
d1['PropertyField4'].fillna('-1', inplace=True)
d1['PropertyField32'].fillna('-1', inplace=True)
d1['PropertyField34'].fillna('-1', inplace=True)
d1['PropertyField36'].fillna('-1', inplace=True)
d1['PropertyField38'].fillna('-1', inplace=True)

print('*' * 80)

print('reading test...')

d2 = pd.read_csv("../input/test.csv", parse_dates=[1])

print(d2.shape)

d2['year'] = d2['Original_Quote_Date'].dt.year
d2['month'] = d2['Original_Quote_Date'].dt.month
d2['wkday'] = d2['Original_Quote_Date'].dt.dayofweek

del d2['Original_Quote_Date']

d2['PropertyField5'] = np.where(d2['PropertyField5'].isnull(), 'Y', d2['PropertyField5'])
d2['PropertyField30'] = np.where(d2['PropertyField30'].isnull(), 'N', d2['PropertyField30'])

tot_features = d2.columns

missing_data_list = []

for v in tot_features:
    n1 = d2[v].isnull().sum()
    if (n1 > 0):
        print(v, n1)
        missing_data_list.append(v)

print(missing_data_list)

unique_keys = []

for v in missing_data_list:
    w = d2[v].unique()
    unique_keys.append(w)
    
n1 = len(missing_data_list)

d2['PropertyField29'].fillna(-1, inplace=True)

d2['PersonalField7'].fillna('-1', inplace=True)
d2['PersonalField84'].fillna(0, inplace=True)

d2['PropertyField3'].fillna('-1', inplace=True)
d2['PropertyField4'].fillna('-1', inplace=True)
d2['PropertyField32'].fillna('-1', inplace=True)
d2['PropertyField34'].fillna('-1', inplace=True)
d2['PropertyField36'].fillna('-1', inplace=True)
d2['PropertyField38'].fillna('-1', inplace=True)

print('*' * 80)

print("Converting objects into numbers...")

object_features = d1.select_dtypes(include=['object']).columns

for v in object_features:
    LE1 = preprocessing.LabelEncoder()
    LE1.fit( list(d1[v].values) + list(d2[v].values) )
    d1[v] = LE1.transform( list(d1[v].values) )
    d2[v] = LE1.transform( list(d2[v].values) )
    d1[v] = d1[v].astype('float')
    d2[v] = d2[v].astype('float')
    print('object',v,'completed...',sep=' ')

d1['QuoteConversion_Flag'] = y

del y

rng = np.random.RandomState(31337)

n10 = len(d1.columns)

xx = np.array(np.arange(1,n10-1),dtype='int64')

train1 = d1

train1.data = d1[ xx ]
train1.target = d1[ [n10-1] ]
train1.feature_names = d1.columns[ xx ]

# split train-test
X_train1, X_test, y_train1, y_test = train_test_split(train1.data,
                                                    train1.target,
                                                    test_size=0.04,
                                                    random_state=1)
                                                    
# split train-calibration
                                                    
X_train, X_cal, y_train, y_cal = train_test_split(X_train1,
                                                    y_train1,
                                                    test_size=0.04,
                                                    random_state=1)


names = feature_names = d1.columns[ xx ]

print(".......names")
print(names)

print('_' * 80)

print("Parallel Parameter optimization")
          
param_dist = {'bst:max_depth':8,
              'bst:eta':0.90,
              'n_estimators': 600,
              'objective':'binary:logistic',
              'nthread':4,
              'eval_metric':'auc',
              'verbose':True}

if __name__ == "__main__":

    y_train = np.array(y_train).ravel()  #xgb.DMatrix('y_train')
    y_cal = np.array(y_cal).ravel()  #xgb.DMatrix('y_train')
    y_test  = np.array(y_test).ravel()
    
    dtrain = xgb.DMatrix(X_train[names], label=y_train)
    dvalid = xgb.DMatrix(X_cal[names], label=y_cal)
    dtest = xgb.DMatrix(X_test[names], label=y_test)
    
    evallist = [(dvalid,'eval'), (dtrain,'train')]
    early_stopping_rounds = 10 #100
    
    clf = xgb.train(param_dist,
                    dtrain,
                    early_stopping_rounds,
                    evallist)
  
    
print('Proceeding to final calculations...')

print("Going into predicting...")

y_train_pred = clf.predict(dtrain)

y_cal_pred = clf.predict(dvalid)

y_test_pred = clf.predict(dtest)

print(np.corrcoef(np.array(y_cal).ravel(), y_cal_pred))

print(np.corrcoef(np.array(y_test).ravel(), y_test_pred))

y_test_pred_clf = clf.predict(dtest)

fpr_test_clf, tpr_test_clf, _ = roc_curve(y_test, y_test_pred_clf)

y_cal_pred_clf = clf.predict(dvalid)

fpr_cal_clf, tpr_cal_clf, _ = roc_curve(y_cal, y_cal_pred_clf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_cal_clf, tpr_cal_clf, label='cal')
plt.plot(fpr_test_clf, tpr_test_clf, label='test')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

t1 = time()

print((t1-t0)/60,' min')

pickle.dump(clf, open("xgb_clf_fit_01a.pkl", "wb"))

names0 = d2.columns

names1 = names0[1:]

for v in names1:
  d2[v] = d2[v].astype('float')

feature_names = names1

print('feature_names')
print(feature_names)

print('*' * 80)
print('loading model...')
#bst1 = pickle.load(open("xgb_clf_fit_01a.pkl", "rb"))

print('predicting...')
d2_test = xgb.DMatrix(d2[feature_names])
preds1 = clf.predict(d2_test)
preds = preds1

result = pd.DataFrame({'QuoteNumber': pd.Series(d2['QuoteNumber']),
                       'QuoteConversion_Flag': pd.Series(preds)},
                       columns=['QuoteNumber','QuoteConversion_Flag'])

print('writing to file...')                       
result.to_csv('draft0712.csv', index=False)
print('time elapsed=',(time()-t0)/60,'min',sep=' ')

print(chr(7) * 3)

plt.show()



