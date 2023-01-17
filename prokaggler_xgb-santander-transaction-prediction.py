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

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler,StandardScaler

import xgboost as xgb

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
df_raw = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv',low_memory=False)
df_raw.head()
X=df_raw.drop('target',axis=1)

y=df_raw.target
df_raw.isnull().sum()/len(df_raw)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=123)
X_train['ID_code'] = X_train['ID_code'].apply(lambda x : str(x).split('_')[1]).copy()
X_train.head()
sc = StandardScaler()

X_train=sc.fit_transform(X_train)

X_train
X_test['ID_code'] = X_test['ID_code'].apply(lambda x : str(x).split('_')[1]).copy()
X_test['ID_code'] = X_test['ID_code'].apply(lambda x : str(x).split('_')[1])
X_test=sc.fit_transform(X_test)

X_test
y_train.head()
xgb = xgb.XGBClassifier()

%timeit xgb.fit(X_train,y_train)

print(xgb)
predict = xgb.predict(X_test)

predict
sample = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sample.head()
accuracy_score(y_test,predict)
roc_auc_score(y_test,predict)
xgb_fine_tune=xgb.XGBClassifier(learning_rate =0.1,

 n_estimators=1000,

n_jobs=-1,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

%timeit xgb_fine_tune.fit(X_train,y_train)

roc_auc_score(y_test,predict)