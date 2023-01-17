import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





from sklearn.neighbors import KNeighborsClassifier



from sklearn.linear_model import LogisticRegression



from sklearn.svm import SVC



from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

sample_submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
test.head()
sample_submission.sample()
train.sample(3)
%%time

train.describe()
train.info(memory_usage='deep')

print('---------')

test.info(memory_usage='deep')
def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)

gl_int = train.select_dtypes(include=['int'])

gl_int2 = test.select_dtypes(include=['int'])

converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')

converted_int2 = gl_int2.apply(pd.to_numeric,downcast='unsigned')

print(mem_usage(gl_int))

print(mem_usage(converted_int))

print('-------------')

print(mem_usage(gl_int2))

print(mem_usage(converted_int2))

compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)

compare_ints.columns = ['before','after']

compare_ints.apply(pd.Series.value_counts)

gl_float = train.select_dtypes(include=['float'])

gl_float2 = test.select_dtypes(include=['float'])

converted_float = gl_float.apply(pd.to_numeric,downcast='float')

converted_float2 = gl_float2.apply(pd.to_numeric,downcast='float')

print(mem_usage(gl_float))

print(mem_usage(converted_float))



print('----------------')



print(mem_usage(gl_float2))

print(mem_usage(converted_float2))



compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)

compare_floats.columns = ['before','after']

compare_floats.apply(pd.Series.value_counts)
tr = train.copy()

tr[converted_int.columns] = converted_int

tr[converted_float.columns] = converted_float

print(mem_usage(train))

print(mem_usage(tr))
%%time

te = test.copy()

te[converted_float.columns] = converted_float

print(mem_usage(test))

print(mem_usage(te))
%%time

f,ax=plt.subplots(1,2, figsize=(12,6))

tr.target.value_counts().plot.pie(explode=[0,0.12],autopct='%1.3f%%',ax=ax[0])

sns.countplot('target',data=tr)

plt.show()

%%time

trainP = tr.hist(figsize = (30,30))

tr.corr()
import lightgbm as lgb

from sklearn.metrics import f1_score, classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold
tr.describe()
tr.drop(['ID_code'], axis=1, inplace=True)

tr.head()

X = tr.iloc[:,1:]

y = tr.iloc[:,0]

from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier()

model.fit(X,y)



feat_importances = pd.Series(model.feature_importances_, index=X.columns)



feat_importances.nlargest(200).plot(kind='barh', figsize=(10,25))

plt.show()
tr.drop(feat_importances.nsmallest(170).keys(), axis=1, inplace=True)
tr.shape
tr_30000 = tr.sample(n=30000, random_state=0)

tr_1500 = tr.sample(n=1500, random_state=0)



#get correlations of each features in dataset

top_corr_features = tr_30000.corr().index

#plot heat map

plt.figure(figsize=(20,20))

g=sns.heatmap(tr_30000[top_corr_features].corr(),annot=True,cmap="YlGnBu")

from sklearn.model_selection import train_test_split
X = tr.iloc[:,1:]

y = tr.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1, stratify = y.values)

%%time

logRes = LogisticRegression()

logRes.fit(X_train, y_train)

y_preds_lr = logRes.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

accuracy_score(y_test, y_preds_lr)

te.drop(['ID_code'], axis=1, inplace=True)

te.head()
X_lr = te.iloc



te.drop(feat_importances.nsmallest(170).keys(), axis=1, inplace=True)
te.shape
y_preds_lr_final = logRes.predict(te)
i = 0

rows_list = []

for pred in y_preds_lr_final:

    row = {'ID_code': 'test_' + str(i), 'target': pred}

    i += 1

    rows_list.append(row)

    

df = pd.DataFrame(rows_list) 

df
df.to_csv("lr_submission.csv", index=False)
dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier()

dtc.fit(X_train, y_train)

rfc.fit(X_train, y_train)

dtc_y_preds = dtc.predict(X_test)

rfc_y_preds = rfc.predict(X_test)
dtc_y_preds = dtc.predict(te)

rfc_y_preds = rfc.predict(te)
i = 0

rows_list = []

for pred in dtc_y_preds:

    row = {'ID_code': 'test_' + str(i), 'target': pred}

    i += 1

    rows_list.append(row)

    

df = pd.DataFrame(rows_list) 

df
df.to_csv("dt_submission.csv", index=False)
i = 0

rows_list = []

for pred in rfc_y_preds:

    row = {'ID_code': 'test_' + str(i), 'target': pred}

    i += 1

    rows_list.append(row)

    

df = pd.DataFrame(rows_list) 

df
df.to_csv("rf_submission.csv", index=False)
import xgboost as xgb

xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed = 123)

xg_cl.fit(X_train, y_train)

y_preds_xgb = xg_cl.predict(te)

y_preds_xgb
i = 0

rows_list = []

for pred in y_preds_xgb:

    row = {'ID_code': 'test_' + str(i), 'target': pred}

    i += 1

    rows_list.append(row)

    

df = pd.DataFrame(rows_list) 

df
df.to_csv("xg_submission.csv", index=False)