# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve

from sklearn.preprocessing import StandardScaler , Binarizer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import time

import os, sys, gc, warnings, random, datetime

import math

import lightgbm as lgb

from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
df = pd.read_pickle("/kaggle/input/handling-imbalanced-data-eda-small-fe/df_for_use.pkl")
plt.figure(figsize = (9,9))

corr = df.corr()

sns.heatmap(corr, cmap='RdBu')
X = df.drop('loan_condition_cat', axis=1)

y = df['loan_condition_cat']





X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)
### LightGBM without Outlier Elimination





start = time.time()



lgbm_clf = LGBMClassifier(n_estimators = 3000, random_state = 2020)

evals = [(X_test, y_test)]

lgbm_clf.fit(X_train, y_train, early_stopping_rounds = 100, eval_metric = 'auc' , eval_set = evals, verbose = 50)

lgbm_cpu_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], average = 'macro')



lgbm_cpu_runtime = time.time() - start



print( 'LightGBM_cpu_ROC_AUC : {0:.4f} , Runtime : {1:.4f}'.format(lgbm_cpu_roc_score ,lgbm_cpu_runtime ))
def get_outlier(df= None, column = None, weight = 5.0):

    #Extract column data with Bad Loan only, get 1/4 percentile and 3/4 percentile through np.percentile

    

    bad_loan = df[df['loan_condition_cat']==1][column]

    quantile_25 = np.percentile(bad_loan.values,25)

    quantile_75 = np.percentile(bad_loan.values,75)

    

    #calculate IQR, multiply with 3, get min,max value

    

    iqr = quantile_75  - quantile_25

    iqr_weight = iqr*weight

    lowest_val = quantile_25 - iqr_weight

    highest_val = quantile_25 + iqr_weight

    

    #fix outlier which is bigger than max, smaller than min

    

    outlier_index = bad_loan[(bad_loan < lowest_val) | (bad_loan > highest_val)].index

    return outlier_index
outlier_index = get_outlier (df = df , column = 'recoveries', weight = 5.0)

print ( "Outlier index :", outlier_index)
def get_preprocessed_df(df=None):

    df_copy = df.copy()

    amount_n = np.log1p(df['loan_amount'])

    df_copy.insert(0, 'Amount_Scaled', amount_n)

    df_copy.drop(['loan_amount'], axis=1, inplace=True)

    # 이상치 데이터 삭제하는 로직 추가

    outlier_index = get_outlier(df=df_copy, column='recoveries', weight=5.0)

    df_copy.drop(outlier_index, axis=0, inplace=True)

    return df_copy

df_copy = get_preprocessed_df(df)
X = df_copy.drop('loan_condition_cat', axis=1)

y = df_copy['loan_condition_cat']





X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)


start = time.time()



lgbm_clf = LGBMClassifier(n_estimators = 3000, random_state = 2020)

evals = [(X_test, y_test)]

lgbm_clf.fit(X_train, y_train, early_stopping_rounds = 100, eval_metric = 'auc' , eval_set = evals, verbose = 50)

lgbm_outlier_eliminated_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], average = 'macro')



lgbm_outlier_eliminated_runtime = time.time() - start



print( 'LightGBM_outlier_eliminated_ROC_AUC : {0:.4f} , Runtime : {1:.4f}'.format(lgbm_outlier_eliminated_roc_score ,lgbm_outlier_eliminated_runtime ))
print( 'LightGBM_cpu_ROC_AUC : {0:.4f} , Runtime : {1:.4f}'.format(lgbm_cpu_roc_score ,lgbm_cpu_runtime ))

print( 'LightGBM_outlier_eliminated_ROC_AUC : {0:.4f} , Runtime : {1:.4f}'.format(lgbm_outlier_eliminated_roc_score ,lgbm_outlier_eliminated_runtime ))





### Negative Effect on Model

#### Opinion : Since DATA is not so skewed, so many columns are designated as outlier and removed