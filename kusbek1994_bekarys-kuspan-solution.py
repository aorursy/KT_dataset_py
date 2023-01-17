import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

import pandas_profiling

from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv('../input/test.csv')
df_train["repay_sum"] = df_train[["repay_sep", "repay_aug", "repay_jul", "repay_jun","repay_may", "repay_april"]].sum(axis = 1)/5

df_train["bill_sum"] = df_train[["bill_sep", "bill_aug", "bill_jul", "bill_jun","bill_may", "bill_apr"]].sum(axis = 1)/5

df_train["pay_sum"] = df_train[["pay_sep", "pay_aug", "pay_jul", "pay_jun","pay_may", "pay_apr"]].sum(axis = 1)/5
df_train["target"].value_counts()
plt.figure(figsize=(20,12))

df_train[df_train["target"]==1]['repay_sum'].hist(alpha=0.5,color='blue',

                                              bins=10,label='target=1')

df_train[df_train["target"]==0]['repay_sum'].hist(alpha=0.5,color='red',

                                              bins=10,label='target=0')

plt.legend()
plt.figure(figsize=(20,12))

df_train[df_train["target"]==1]['credit_bal'].hist(alpha=0.5,color='blue',

                                              bins=20,label='target=1')

df_train[df_train["target"]==0]['credit_bal'].hist(alpha=0.5,color='red',

                                              bins=20,label='target=0')

plt.legend()
plt.figure(figsize=(20,12))

df_train[df_train["target"]==1]['bill_sum'].hist(alpha=0.5,color='blue',

                                              bins=20,label='target=1')

df_train[df_train["target"]==0]['bill_sum'].hist(alpha=0.5,color='red',

                                              bins=20,label='target=0')

plt.legend()
plt.figure(figsize=(20,12))

df_train[df_train["target"]==1]['repay_sum'].hist(alpha=0.5,color='blue',

                                              bins=20,label='target=1')

df_train[df_train["target"]==0]['repay_sum'].hist(alpha=0.5,color='red',

                                              bins=20,label='target=0')

plt.legend()
y = df_train["target"]

df_train.drop(["id_code", "target"], axis =1, inplace=True)

X = pd.get_dummies(df_train)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state = 101)
rf = RandomForestClassifier(bootstrap=True, max_depth = 100, max_features = 3, min_samples_leaf=4, min_samples_split=8, n_estimators=300, random_state=101)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred_rf)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#df_test["pay_sum"] = df_test[["pay_sep", "pay_aug", "pay_jul", "pay_jun","pay_may", "pay_apr"]].sum(axis = 1)

#df_test["bill_sum"] = df_test[["bill_sep", "bill_aug", "bill_jul", "bill_jun","bill_may", "bill_apr"]].sum(axis = 1)

#df_test["repay_sum"] = df_test[["repay_sep", "repay_aug", "repay_jul", "repay_jun","repay_may", "repay_april"]].sum(axis = 1)

#df_test.drop(["id_code"], axis =1, inplace=True)

#X_test_test = pd.get_dummies(df_test)

#y_pred_test = rf.predict_proba(X_test_test)[:,1]

#sample = pd.read_csv("../input/sample_submission.csv")

#sample["target"] = y_pred_test

#sample.to_csv("rf_submission.csv", index=False)
from sklearn.ensemble import GradientBoostingClassifier

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in learning_rates:

    gb = GradientBoostingClassifier(n_estimators=400, learning_rate = learning_rate, max_features=4, max_depth = 3, random_state = 101)

    gb.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)

    y_pred_gb = gb.predict_proba(X_test)[:,1]

    print("Accuracy score (validation): {0:.3f}".format(roc_auc_score(y_test, y_pred_gb)))

    print()
#from sklearn.model_selection import GridSearchCV

#param_grid = {

#    'learning_rate' : [0.05, 0.1, 0.25, 0.5, 0.75, 1],

#    'max_depth': [80, 90, 100, 110],

#    'max_features': [2, 3, 4],

#    'min_samples_split': [2, 3, 4],

#    'n_estimators': [100, 200]

#    }

#gb = GradientBoostingClassifier()

#grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, 

#                          cv = 7, n_jobs = -1, verbose = 2)

#grid_search.fit(X_train, y_train)

#grid_search.best_params_
from sklearn.model_selection import cross_val_score

gb = GradientBoostingClassifier(n_estimators=400, learning_rate = 0.05, max_features=5, max_depth = 3, random_state = 101)

scores = cross_val_score(gb, X_train, y_train, cv=10, scoring='roc_auc')

print(scores)
gb = GradientBoostingClassifier(n_estimators=400, learning_rate = 0.05, max_features=5, max_depth = 3, random_state = 101)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred_gb)
df_test["pay_sum"] = df_test[["pay_sep", "pay_aug", "pay_jul", "pay_jun","pay_may", "pay_apr"]].sum(axis = 1)

df_test["bill_sum"] = df_test[["bill_sep", "bill_aug", "bill_jul", "bill_jun","bill_may", "bill_apr"]].sum(axis = 1)

df_test["repay_sum"] = df_test[["repay_sep", "repay_aug", "repay_jul", "repay_jun","repay_may", "repay_april"]].sum(axis = 1)

df_test.drop(["id_code"], axis =1, inplace=True)

X_test_test = pd.get_dummies(df_test)

y_pred_test = gb.predict_proba(X_test_test)[:,1]

sample = pd.read_csv("../input/sample_submission.csv")

sample["target"] = y_pred_test

sample.to_csv("gb_submission.csv", index=False)
import numpy as np

import pandas as pd

import pandas.core.algorithms as algos

from pandas import Series

import scipy.stats.stats as stats

import re

import traceback

import string



max_bin = 20

force_bin = 3



def mono_bin(Y, X, n = max_bin):

    

    df1 = pd.DataFrame({"X": X, "Y": Y})

    justmiss = df1[['X','Y']][df1.X.isnull()]

    notmiss = df1[['X','Y']][df1.X.notnull()]

    r = 0

    while np.abs(r) < 1:

        try:

            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})

            d2 = d1.groupby('Bucket', as_index=True)

            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)

            n = n - 1 

        except Exception as e:

            n = n - 1



    if len(d2) == 1:

        n = force_bin         

        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))

        if len(np.unique(bins)) == 2:

            bins = np.insert(bins, 0, 1)

            bins[1] = bins[1]-(bins[1]/2)

        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 

        d2 = d1.groupby('Bucket', as_index=True)

    

    d3 = pd.DataFrame({},index=[])

    d3["MIN_VALUE"] = d2.min().X

    d3["MAX_VALUE"] = d2.max().X

    d3["COUNT"] = d2.count().Y

    d3["EVENT"] = d2.sum().Y

    d3["NONEVENT"] = d2.count().Y - d2.sum().Y

    d3=d3.reset_index(drop=True)

    

    if len(justmiss.index) > 0:

        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])

        d4["MAX_VALUE"] = np.nan

        d4["COUNT"] = justmiss.count().Y

        d4["EVENT"] = justmiss.sum().Y

        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y

        d3 = d3.append(d4,ignore_index=True)

    

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT

    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT

    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT

    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT

    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["VAR_NAME"] = "VAR"

    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       

    d3 = d3.replace([np.inf, -np.inf], 0)

    d3.IV = d3.IV.sum()

    

    return(d3)



def char_bin(Y, X):

        

    df1 = pd.DataFrame({"X": X, "Y": Y})

    justmiss = df1[['X','Y']][df1.X.isnull()]

    notmiss = df1[['X','Y']][df1.X.notnull()]    

    df2 = notmiss.groupby('X',as_index=True)

    

    d3 = pd.DataFrame({},index=[])

    d3["COUNT"] = df2.count().Y

    d3["MIN_VALUE"] = df2.sum().Y.index

    d3["MAX_VALUE"] = d3["MIN_VALUE"]

    d3["EVENT"] = df2.sum().Y

    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    

    if len(justmiss.index) > 0:

        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])

        d4["MAX_VALUE"] = np.nan

        d4["COUNT"] = justmiss.count().Y

        d4["EVENT"] = justmiss.sum().Y

        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y

        d3 = d3.append(d4,ignore_index=True)

    

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT

    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT

    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT

    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT

    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3["VAR_NAME"] = "VAR"

    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      

    d3 = d3.replace([np.inf, -np.inf], 0)

    d3.IV = d3.IV.sum()

    d3 = d3.reset_index(drop=True)

    

    return(d3)



def data_vars(df1, target):

    

    stack = traceback.extract_stack()

    filename, lineno, function_name, code = stack[-2]

    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]

    final = (re.findall(r"[\w']+", vars_name))[-1]

    

    x = df1.dtypes.index

    count = -1

    

    for i in x:

        if i.upper() not in (final.upper()):

            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:

                conv = mono_bin(target, df1[i])

                conv["VAR_NAME"] = i

                count = count + 1

            else:

                conv = char_bin(target, df1[i])

                conv["VAR_NAME"] = i            

                count = count + 1

                

            if count == 0:

                iv_df = conv

            else:

                iv_df = iv_df.append(conv,ignore_index=True)

    

    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})

    iv = iv.reset_index()

    return(iv_df,iv)

dfiv,iv=data_vars(df_train.drop(["id_code", "target"], axis =1),df_train["target"])
dfiv
iv.sort_values(by="IV", ascending = False)