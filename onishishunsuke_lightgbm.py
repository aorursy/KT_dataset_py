# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

import category_encoders as ce



from sklearn.preprocessing import StandardScaler



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/train.csv')

df_test = pd.read_csv('/kaggle/input/test.csv')

df_country = pd.read_csv('/kaggle/input/country_info.csv')



#submit用データ

df_survey = pd.read_csv('/kaggle/input/survey_dictionary.csv', index_col=0)
X_train = df_train.loc[:,['Employment','AssessJob1','Age']]

y_train = df_train['ConvertedSalary']



X_test = df_test.loc[:,['Employment','AssessJob1','Age']]
print(df_train['Gender'].value_counts(dropna=False))
print(df_train['Country'].value_counts(dropna=False)),print(df_test['Country'].value_counts(dropna=False))
print(df_train['CompanySize'].value_counts(dropna=False)),print(df_test['CompanySize'].value_counts(dropna=False))
nu_col = df_train.nunique()

print(nu_col)

print(type(nu_col))
nu_col = df_test.nunique()

print(nu_col)

print(type(nu_col))
X_train.AssessJob1.fillna(-99,inplace=True)



# test data

X_test.AssessJob1.fillna(-99,inplace=True)

# 文字列抽出

cats = []

for col in X_train.columns:

        if X_train[col].dtype == 'object':

            cats.append(col)

            

            print(col, X_train[col].nunique())
Employment_map = {'Employed full-time':5,

                  'Employed part-time':4,

                  'Independent contractor, freelancer, or self-employed':6,

                  'Not employed, but looking for work':3,

                  'Not employed, and not looking for work':2,

                  'Retired':1

                  }



X_train['Employment'] = X_train['Employment'].map(Employment_map)

X_test['Employment'] = X_train['Employment'].map(Employment_map)



Age_map = {'Under 18 years old':1, '18 - 24 years old':2,

           '25 - 34 years old':3, '35 - 44 years old':4,

           '45 - 54 years old':5, '55 - 64 years old':6, '65 years or older':7

           }



X_train['Age'] = X_train['Age'].map(Age_map)

X_test['Age'] = X_train['Age'].map(Age_map)

X_train['train']=1

X_test['train']=0
# 文字列のエンコード

X_all = pd.concat([X_train, X_test])

oe = ce.OrdinalEncoder(cols=cats,handle_unknown='impute',return_df=False)



X_all[cats] = oe.fit_transform(X_all[cats])



X_train = X_all[X_all['train']==1]

X_test = X_all[X_all['train']==0]
X_train = X_train.drop(['train'], axis=1) 

X_test = X_test.drop(['train'], axis=1) 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
import lightgbm as lgb

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt



plt.style.use('ggplot')

%matplotlib inline



scores = []



skf = StratifiedKFold(n_splits=2, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #scikitlearnのLightGBM

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8, #colsample_bytree：1木あたりで使う項目数（割合）

                         importance_type='split', learning_rate=0.05, max_depth=-1,

                         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                         n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None, #n_estimators（木の数）は最大にしておいてearlyStoppingに任せる

                         random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                         subsample=0.9, subsample_for_bin=200000, subsample_freq=0) #lightGBMは連続値もbinningする。



    clf.fit(X_train_, y_train_, early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

   

    y_pred_2 = clf.predict_proba(X_test)[:,1]

    submit['ConvertedSalary'] = y_pred_2

    submit_lgb =  pd.merge(submit_lgb, submit, left_on=['Respondent'], right_on=['Respondent'], how='left')



    score = roc_auc_score(y_val, y_pred)

    print(score)



    fig, ax = plt.subplots(figsize=(10, 15))

    lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')