# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/preliminary-feature-engineered-manually/data_feature_engineered.csv')
data.head().T
data.shape
label_counts = data['Target'].value_counts().sort_index()

label_counts
data1 = data.values
data1[0:5]
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data1[:,6] = labelencoder.fit_transform(data1[:,6])

data1[:,7] = labelencoder.fit_transform(data1[:,7])

data1[:,8] = labelencoder.fit_transform(data1[:,8])

data1[:,10] = labelencoder.fit_transform(data1[:,10])

data1[:,11] = labelencoder.fit_transform(data1[:,11])

data1[:,12] = labelencoder.fit_transform(data1[:,12])

data1[:,13] = labelencoder.fit_transform(data1[:,13])

data1[:,14] = labelencoder.fit_transform(data1[:,14])

data1[:,15] = labelencoder.fit_transform(data1[:,15])

data1[:,16] = labelencoder.fit_transform(data1[:,16])

data1[:,17] = labelencoder.fit_transform(data1[:,17])

data1[:,24] = labelencoder.fit_transform(data1[:,24])

data1[:,32] = labelencoder.fit_transform(data1[:,32])

data1[:,33] = labelencoder.fit_transform(data1[:,33])
data1[0:5]
data2 = pd.DataFrame(data1)
data2.shape
from sklearn.utils import resample
is_1 =  data2[35]==1

is_1[0:10]
data_1 = data2[is_1]

print(data_1.shape)
data_1.head().T
data_11 = resample(data_1,replace=True,n_samples=1900,random_state=10)
data_11.shape
is_2 =  data2[35]==2

data_2 = data2[is_2]

print(data_2.shape)

data_22 = resample(data_2,replace=True,n_samples=1900,random_state=10)

print(data_22.shape)
data_22.head()
is_3 =  data2[35]==3

data_3 = data2[is_3]

print(data_3.shape)

data_33 = resample(data_3,replace=True,n_samples=1900,random_state=10)

print(data_33.shape)
is_4 =  data2[35]==4

data_4 = data2[is_4]

print(data_4.shape)

data_44 = resample(data_4,replace=False,n_samples=700,random_state=10)

print(data_44.shape)
data_new=np.vstack((data_11,data_22,data_33,data_4))
data_new.shape
data_new[0:5]
X = data_new[:,1:35]
X.shape
X[0:5]
y = data_new[:,35]
y[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=100)
y_train = y_train.astype('int')

y_test = y_test.astype('int')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = RandomForestClassifier(max_features='sqrt',min_samples_leaf=4,n_estimators = 1000,n_jobs=-1,oob_score = True,

                             random_state=10,warm_start = False)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
import xgboost as xgb
xg_clf = xgb.XGBClassifier(colsample_bytree = 0.7, learning_rate = 0.1,

                max_depth = 30, alpha = 10, n_estimators = 300)
xg_clf.fit(X_train,y_train)

y_preds = xg_clf.predict(X_test)
accuracy_score(y_preds,y_test)
y_preds[0:5]
y_test[0:5]
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy

rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')

boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)

boruta_selector.fit(X_train,y_train)

print("==============BORUTA==============")

print (boruta_selector.n_features_)
print(boruta_selector.ranking_)
X_train.shape
X1 = X_train[:,0]

X2 = X_train[:,2]

X3 = X_train[:,5]

X4 = X_train[:,12]

X5 = X_train[:,15]

X6 = X_train[:,17]

X7 = X_train[:,18]

X8 = X_train[:,22]

X9 = X_train[:,20]

X10 = X_train[:,21]

X11 = X_train[:,33]

X12 = X_train[:,30]

X13 = X_train[:,31]



Xt1 = X_test[:,0]

Xt2 = X_test[:,2]

Xt3 = X_test[:,5]

Xt4 = X_test[:,12]

Xt5 = X_test[:,15]

Xt6 = X_test[:,17]

Xt7 = X_test[:,18]

Xt8 = X_test[:,22]

Xt9 = X_test[:,20]

Xt10 = X_test[:,21]

Xt11 = X_test[:,33]

Xt12 = X_test[:,30]

Xt13 = X_test[:,31]
X_train_boruta = np.vstack((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13))

X_test_boruta = np.vstack((Xt1,Xt2,Xt3,Xt4,Xt5,Xt6,Xt7,Xt8,Xt9,Xt10,Xt11,Xt12,Xt13))
X_train_boruta = pd.DataFrame(X_train_boruta)

X_train_boruta = X_train_boruta.T

X_train_boruta = X_train_boruta.values
X_test_boruta = pd.DataFrame(X_test_boruta)

X_test_boruta = X_test_boruta.T

X_test_boruta = X_test_boruta.values
X_test_boruta.shape
opt_parameters = {'max_depth':35, 'eta':0.15, 'silent':1, 'objective':'multi:softmax', 'min_child_weight': 2, 'num_class': 4, 'gamma': 2.5, 'colsample_bylevel': 1, 'subsample': 0.95, 'colsample_bytree': 0.85, 'reg_lambda': 0.35 }
def evaluate_macroF1_lgb(predictions, truth):  

    pred_labels = predictions.argmax(axis=1)

    truth = truth.get_label()

    f1 = f1_score(truth, pred_labels, average='macro')

    return ('macroF1', 1-f1) 



fit_params={"early_stopping_rounds":500,

            "eval_metric" : evaluate_macroF1_lgb, 

            "eval_set" : [(X_train_boruta,y_train), (X_test_boruta,y_test)],

            'verbose': False,

           }



def learning_rate_power_0997(current_iter):

    base_learning_rate = 0.1

    min_learning_rate = 0.02

    lr = base_learning_rate  * np.power(.995, current_iter)

    return max(lr, min_learning_rate)



fit_params['verbose'] = 50
from sklearn.ensemble import VotingClassifier

clfs = []

for i in range(15):

    clf = xgb.XGBClassifier(random_state=217+i, n_estimators=300, learning_rate=0.15, n_jobs=4, **opt_parameters)

    

    clfs.append(('xgb{}'.format(i), clf))

    

vc = VotingClassifier(clfs, voting='soft')

del(clfs)



_ = vc.fit(X_train_boruta, y_train)



clf_final = vc.estimators_[0]
#vc.fit(X_train_boruta,y_train)

y_preds = vc.predict(X_test_boruta)

accuracy_score(y_preds,y_test)
xg_clf.fit(X_train_boruta,y_train)

y_preds = xg_clf.predict(X_test_boruta)

accuracy_score(y_preds,y_test)
testdata = pd.read_csv('../input/test-file/test_feature_engineered.csv')
testdata.head().T
testdata.shape
testdata1 = testdata.values
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

testdata1[:,6] = labelencoder.fit_transform(testdata1[:,6])

testdata1[:,7] = labelencoder.fit_transform(testdata1[:,7])

testdata1[:,8] = labelencoder.fit_transform(testdata1[:,8])

testdata1[:,10] = labelencoder.fit_transform(testdata1[:,10])

testdata1[:,11] = labelencoder.fit_transform(testdata1[:,11])

testdata1[:,12] = labelencoder.fit_transform(testdata1[:,12])

testdata1[:,13] = labelencoder.fit_transform(testdata1[:,13])

testdata1[:,14] = labelencoder.fit_transform(testdata1[:,14])

testdata1[:,15] = labelencoder.fit_transform(testdata1[:,15])

testdata1[:,16] = labelencoder.fit_transform(testdata1[:,16])

testdata1[:,17] = labelencoder.fit_transform(testdata1[:,17])

testdata1[:,24] = labelencoder.fit_transform(testdata1[:,24])

testdata1[:,32] = labelencoder.fit_transform(testdata1[:,32])

testdata1[:,33] = labelencoder.fit_transform(testdata1[:,33])
testdata1 = testdata1[:,1:35]
Xtd1 = testdata1[:,0]

Xtd2 = testdata1[:,2]

Xtd3 = testdata1[:,5]

Xtd4 = testdata1[:,12]

Xtd5 = testdata1[:,15]

Xtd6 = testdata1[:,17]

Xtd7 = testdata1[:,18]

Xtd8 = testdata1[:,22]

Xtd9 = testdata1[:,20]

Xtd10 = testdata1[:,21]

Xtd11 = testdata1[:,33]

Xtd12 = testdata1[:,30]

Xtd13 = testdata1[:,31]
X_testdata_boruta = np.vstack((Xtd1,Xtd2,Xtd3,Xtd4,Xtd5,Xtd6,Xtd7,Xtd8,Xtd9,Xtd10,Xtd11,Xtd12,Xtd13))
X_testdata_boruta = pd.DataFrame(X_testdata_boruta)

X_testdata_boruta = X_testdata_boruta.T

X_testdata_boruta = X_testdata_boruta.values
X_testdata_boruta.shape
y_preds_test = xg_clf.predict(X_testdata_boruta)
y_preds_test[0:20]
testdata['label'] = y_preds_test

testdata.to_csv('result_costa_rica.csv', index=False)
# used vlookup to complete the test file given from our result table.
y_preds_test2 = vc.predict(X_testdata_boruta)
y_preds_test2[0:20]
#testdata['label'] = y_preds_test2

#testdata.to_csv('result_costa_rica_vc.csv', index=False)
result_file = pd.read_csv('../input/submission12/submisionvc.csv')

result_file.to_csv('submission.csv',index=False)