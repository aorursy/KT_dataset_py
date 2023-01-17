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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import category_encoders as ce

#from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import OrdinalEncoder



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm



from sklearn.feature_extraction.text import TfidfVectorizer



import lightgbm as lgb

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline
# kaggle上のtrain.csvを「df_train」として読み込む。0列目のcol（id）がindex

df_train = pd.read_csv('../input/train.csv', index_col=0)



# kaggle上のtest.csvを「df_test」として読み込む。0列目のcol（id）がindex

df_test = pd.read_csv('../input/test.csv', index_col=0)



# kaggle上のcountry_info.csvを「df_test」として読み込む。0列目のcol（id）がindex

country_info = pd.read_csv('../input/country_info.csv', index_col=0)
#df_train = df_train[df_train['Country'].str.contains('United Kingdom')]
y_train = df_train.ConvertedSalary

x_train = df_train.drop(['ConvertedSalary'], axis=1)

x_test = df_test
x_train = pd.merge(x_train, country_info, on='Country', how='left')

x_test = pd.merge(x_test, country_info, on='Country', how='left')
cats = []

for col in x_train.columns:

    if x_train[col].dtype == 'object':

        cats.append(col)

        print(col,x_train[col].nunique())
oe = ce.OrdinalEncoder(cols=cats, return_df=False)



x_train[cats] = oe.fit_transform(x_train[cats])

x_test[cats] = oe.transform(x_test[cats])
# 全体の処理



# 中央値で埋めた「x_train.median()」を新たなx_trainにする

x_train=x_train.fillna(x_train.median())



# 中央値で埋めた「x_test.median()」を新たなx_testにする

x_test=x_test.fillna(x_test.median())



# 0埋め



#x_train=x_train.fillna(0)

#x_test=x_test.fillna(0)
# XGBoost

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(x_train, y_train)

pred = pd.DataFrame( {'XGB': reg_xgb.predict(x_test)})





# 作ったモデルにx_testを入れて予測をする。これをpredと呼ぶ

#pred = reg_xgb.predict_proba(x_test)[:,1]
# kaggle上のsample_submission.csvを呼び出して（0colがindex）、submissionyyyymmdd_v??とする

submission20190807_01_xgb = pd.read_csv('../input/sample_submission.csv', index_col=0)
# submissionyyyymmdd_v??にpredのloan_conditionを入れる

submission20190807_01_xgb['ConvertedSalary'] = pred
# submissionyyyymmdd_v??をkaggle上に保存

submission20190807_01_xgb.to_csv('./submission20190807_01.csv')