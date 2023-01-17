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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

app = pd.read_csv('../input/applications/applications.csv')
app['CustomerID'].drop_duplicates().shape

first_app = app.iloc[app['CustomerID'].drop_duplicates().index]
train.set_index('TZ',inplace=True)

test.set_index('TZ',inplace=True)

first_app.set_index('CustomerID',inplace=True)
train = train.join(first_app,how='left')

test = test.join(first_app,how='left')
y = train['NESHER']
features = train.drop(['Begin_date','End_date','NESHER','MISPAR_ISHI','DESTINATION','MAHZOR_ACHARON'],axis=1).columns
# Categorical boolean mask

categorical_feature_mask = train[features].dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = train[features].columns[categorical_feature_mask].tolist()
train[categorical_cols] = train[categorical_cols].astype(str)

test[categorical_cols] = test[categorical_cols].astype(str)
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()
# apply le on categorical feature columns

train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col))

train[categorical_cols].head(10)
# apply le on categorical feature columns

test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col))
X_train = train[features][:15000]

y_train = y[:15000]



X_val = train[features][15000:]

y_val = y[15000:]
import lightgbm as lgb



model = lgb.LGBMClassifier()



model.fit(X_train,y_train)
from sklearn.metrics import log_loss, f1_score, accuracy_score, cohen_kappa_score,roc_auc_score
y_pred = model.predict(X_val)



accuracy_score(y_val, y_pred)
y_pred_prob = model.predict_proba(X_val)[:,1]



roc_auc_score(y_val, y_pred_prob)
SampleSubmission = pd.read_csv('../input/SampleSubmission.csv')



test_p =  model.predict_proba(test[features])[:,1]



sub = pd.DataFrame(np.stack([test.index,test_p], axis=1), columns=SampleSubmission.columns)



sub.to_csv('/kaggle/working/sub.csv', index=False)