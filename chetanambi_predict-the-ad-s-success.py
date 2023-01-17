import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

sns.set_style('darkgrid')

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/Train.csv')

test = pd.read_csv('/kaggle/input/Test.csv')

sub = pd.read_csv('/kaggle/input/sample_submission.csv')
train.shape, test.shape, sub.shape
train.duplicated().sum(), test.duplicated().sum()
train.head(2)
train.info()
train.isnull().sum()
train.nunique()
df = train.append(test, ignore_index=True, sort=False)
df.head(2)
df['ratings_bin'] = pd.qcut(df['ratings'], q=5, labels=False, duplicates='drop')
for col in df.columns:

    if col not in ['id','average_runtime(minutes_per_week)','ratings','netgain']:

        df[col] = df[col].astype('category')
df = pd.get_dummies(data=df, columns=['realtionship_status','industry','genre','targeted_sex','airtime','airlocation','expensive','money_back_guarantee'], drop_first=True)
df.drop(['id'], axis=1, inplace=True)
train_df = df[df['netgain'].isnull()!=True]

test_df = df[df['netgain'].isnull()==True]

test_df.drop('netgain', axis=1, inplace=True)
train_df.shape, test_df.shape
train_df['netgain'] = train_df['netgain'].map({True:1,False:0})
X = train_df.drop(labels=['netgain'], axis=1)

y = train_df['netgain'].values



from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
X_train.head(2)
from math import sqrt 

from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier



param = {'boosting': 'gbdt',  

         'learning_rate': 0.09, 

         'num_iterations': 1000,

         'num_leaves': 80,

         'max_depth': -1,

         'max_bin': 100,

         'min_data_in_leaf': 10,

         'bagging_fraction': 0.90,

         'bagging_freq': 1,

         'feature_fraction': 0.9

         }



lgbm = LGBMClassifier(**param)

lgbm.fit(X_train, y_train, eval_set = [(X_train,y_train),(X_cv, y_cv)], early_stopping_rounds=100, verbose=300)

y_pred_lgbm = lgbm.predict(X_cv)
predictions = []

for x in y_pred_lgbm:

    predictions.append(np.argmax(x))

print('accuracy:', accuracy_score(y_cv, predictions))
Xtest = test_df
from sklearn.model_selection import KFold, StratifiedKFold

from lightgbm import LGBMClassifier



errlgb = []

y_pred_totlgb = []



fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X, y):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    param = {'boosting': 'gbdt',  

         'learning_rate': 0.09, 

         'num_iterations': 1000,

         'num_leaves': 80,

         'max_depth': -1,

         'max_bin': 50,

         'min_data_in_leaf': 10,

         'bagging_fraction': 0.9,

         'bagging_freq': 1,

         'feature_fraction': 0.9

         }

    

    lgbm = LGBMClassifier(**param)

    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=200)



    y_pred_lgbm = lgbm.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test,y_pred_lgbm))



    errlgb.append(accuracy_score(y_test,y_pred_lgbm))

    p = lgbm.predict(Xtest)

    y_pred_totlgb.append(p)
np.mean(errlgb) 
lgbm_final = np.mean(y_pred_totlgb,0).round()

lgbm_final
sub = pd.DataFrame({'id':test.id,'netgain': lgbm_final})

sub.head()
sub['netgain'] = sub['netgain'].map({1:True,0:False})

sub['netgain'].value_counts()
sub.to_csv('Output.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)