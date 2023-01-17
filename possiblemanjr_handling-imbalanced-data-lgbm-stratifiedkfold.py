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

import xgboost as xgb

from time import time

import os, sys, gc, warnings, random, datetime

import math

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold , KFold

# from ngboost import NGBRegressor

import lightgbm as lgb

from lightgbm import LGBMClassifier
df = pd.read_pickle("/kaggle/input/loan-condition-eda-data-cleansing/df_for_use.pkl")
def get_clf_eval(y_test, pred):

    confusion = confusion_matrix(y_test, pred)

    accuracy = accuracy_score(y_test , pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test,pred)

    f1 = f1_score(y_test, pred)

    auc = roc_auc_score(y_test, pred)

    print('Confusion Matrix')

    print(confusion)

    print('Auccuracy : {0:.4f}, Precision : {1:.4f} , Recall : {2:.4f} , F1_Score : {3:.4f}, ROC_AUC_Score : {4:.4f}'.format(accuracy , precision, recall, f1, auc))
thresholds = {0.3,0.35, 0.4, 0.45, 0.50, 0.55, 0.60}



def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):

    for custom_threshold in thresholds:

        binarizer = Binarizer(threshold = custom_threshold).fit(pred_proba_c1)

        custom_predict = binarizer.transform(pred_proba_c1)

        print('threshold:', custom_threshold)

        get_clf_eval(y_test, custom_predict)



## get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
X = df.drop('loan_condition_cat', axis=1)

y = df['loan_condition_cat']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)

import time
### No Fold (Using GPU)



start = time.time()



lgbm_clf = LGBMClassifier( n_estimators = 3000, random_state = 2020)

evals = [(X_test, y_test)]

lgbm_clf.fit(X_train, y_train, early_stopping_rounds = 100, eval_metric = 'auc' , eval_set = evals, verbose = 50)

lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], average = 'macro')

print( 'ROC_AUC : {0:.4f}'.format(lgbm_roc_score))



print("Runtime :", time.time() - start)
X = df.drop('loan_condition_cat', axis=1)

y = df['loan_condition_cat']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)





from lightgbm import LGBMClassifier



from time import time

params_lgb={'boosting_type':'gbdt',

           'objective': 'binary',

           'random_state':2020,

           'metric':'auc'

           }



k_fold=5

kf=StratifiedKFold(n_splits=k_fold,shuffle=True, random_state=2020)

training_start_time = time()

aucs=[]

y_preds = np.zeros(X_test.shape[0])



for fold, (trn_idx,val_idx) in enumerate(kf.split(X_train,y_train)):

    start_time = time()

    print('Training on fold {}'.format(fold + 1))

    trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

    clf = lgb.train(params_lgb, trn_data, num_boost_round=10000, valid_sets = [trn_data, val_data], 

                    verbose_eval=200, early_stopping_rounds=200)

    aucs.append(clf.best_score['valid_1']['auc'])

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    y_preds += clf.predict(X_test) / 5

    

    

    

print('-' * 30)

print('Training is completed!.')

print("\n## Mean CV_AUC_Score : ", np.mean(aucs))

print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))

# print(clf.best_params_)

print('-' * 30)





# pred_rf = clf.predict(X_test)

auc = roc_auc_score(y_test,y_preds)

print(' ROC_AUC_Score : {0:.4f}'.format (auc))
X = df.drop('loan_condition_cat', axis=1)

y = df['loan_condition_cat']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)

from time import time

params_lgb={'boosting_type':'gbdt',

           'objective': 'binary',

           'random_state':2020,

           'metric':'auc'}



k_fold=5

kf=StratifiedKFold(n_splits=k_fold,shuffle=True, random_state=2020)

training_start_time = time()

aucs=[]



for fold, (trn_idx,val_idx) in enumerate(kf.split(X,y)):

    start_time = time()

    print('Training on fold {}'.format(fold + 1))

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

    clf = lgb.train(params_lgb, trn_data, num_boost_round=10000, valid_sets = [trn_data, val_data], 

                    verbose_eval=200, early_stopping_rounds=200)

    aucs.append(clf.best_score['valid_1']['auc'])

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    

print('-' * 30)

print('Training is completed!.')

print("\n## Mean CV_AUC_Score : ", np.mean(aucs))

print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))

# print(clf.best_params_)

print('-' * 30)



# X_test = test_df.drop('loan_condition_cat', axis=1)

# y_test = test_df['loan_condition_cat']



# pred_rf = clf.predict(X_test)

# auc = roc_auc_score(y_test,pred_rf)

# print(' ROC_AUC_Score : {0:.4f}'.format (auc))