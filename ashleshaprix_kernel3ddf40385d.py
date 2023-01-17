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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

#import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial





import gc



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_id = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

df_trans = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def CalcOutliers(df_num): 



    # calculating mean and std of the array

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # seting the cut line to both higher and lower values

    # You can change this value

    cut = data_std * 3



    #Calculating the higher and lower cut values

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers

    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers

    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides

    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values

    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

    

    return
## REducing memory

df_trans = reduce_mem_usage(df_trans)

df_id = reduce_mem_usage(df_id)

resumetable(df_trans)[:25]
df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)

total = len(df_trans)

total_amt = df_trans.groupby(['isFraud'])['TransactionAmt'].sum().sum()

plt.figure(figsize=(18,8))



plt.subplot(121)

g = sns.countplot(x='isFraud', data=df_trans, )

g.set_title("Fraud Transactions Distribution \n# 0: No Fraud | 1: Fraud #", fontsize=22)

g.set_xlabel("Is fraud?", fontsize=18)

g.set_ylabel('Count', fontsize=18)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 



perc_amt = (df_trans.groupby(['isFraud'])['TransactionAmt'].sum())

perc_amt = perc_amt.reset_index()

plt.subplot(122)

g1 = sns.barplot(x='isFraud', y='TransactionAmt',  dodge=True, data=perc_amt)

g1.set_title("% Total Amount in Transaction Amt \n# 0: No Fraud | 1: Fraud #", fontsize=22)

g1.set_xlabel("Is fraud?", fontsize=18)

g1.set_ylabel('Total Transaction Amount Scalar', fontsize=18)

for p in g1.patches:

    height = p.get_height()

    g1.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total_amt * 100),

            ha="center", fontsize=15) 

    

plt.show()
df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)

print("Transaction Amounts Quantiles:")

print(df_trans['TransactionAmt'].quantile([.01, .025, .1, .25, .5, .75, .9, .975, .99]))
plt.figure(figsize=(16,12))

plt.suptitle('Transaction Values Distribution', fontsize=22)

plt.subplot(221)

g = sns.distplot(df_trans[df_trans['TransactionAmt'] <= 1000]['TransactionAmt'])

g.set_title("Transaction Amount Distribuition <= 1000", fontsize=18)

g.set_xlabel("")

g.set_ylabel("Probability", fontsize=15)



plt.subplot(222)

g1 = sns.distplot(np.log(df_trans['TransactionAmt']))

g1.set_title("Transaction Amount (Log) Distribuition", fontsize=18)

g1.set_xlabel("")

g1.set_ylabel("Probability", fontsize=15)



plt.figure(figsize=(16,12))





plt.subplot(212)

g4 = plt.scatter(range(df_trans[df_trans['isFraud'] == 0].shape[0]),

                 np.sort(df_trans[df_trans['isFraud'] == 0]['TransactionAmt'].values), 

                 label='NoFraud', alpha=.2)

g4 = plt.scatter(range(df_trans[df_trans['isFraud'] == 1].shape[0]),

                 np.sort(df_trans[df_trans['isFraud'] == 1]['TransactionAmt'].values), 

                 label='Fraud', alpha=.2)

g4= plt.title("ECDF \nFRAUD and NO FRAUD Transaction Amount Distribution", fontsize=18)

g4 = plt.xlabel("Index")

g4 = plt.ylabel("Amount Distribution", fontsize=15)

g4 = plt.legend()



plt.figure(figsize=(16,12))



plt.subplot(321)

g = plt.scatter(range(df_trans[df_trans['isFraud'] == 1].shape[0]), 

                 np.sort(df_trans[df_trans['isFraud'] == 1]['TransactionAmt'].values), 

                label='isFraud', alpha=.4)

plt.title("FRAUD - Transaction Amount ECDF", fontsize=18)

plt.xlabel("Index")

plt.ylabel("Amount Distribution", fontsize=12)



plt.subplot(322)

g1 = plt.scatter(range(df_trans[df_trans['isFraud'] == 0].shape[0]),

                 np.sort(df_trans[df_trans['isFraud'] == 0]['TransactionAmt'].values), 

                 label='NoFraud', alpha=.2)

g1= plt.title("NO FRAUD - Transaction Amount ECDF", fontsize=18)

g1 = plt.xlabel("Index")

g1 = plt.ylabel("Amount Distribution", fontsize=15)



plt.suptitle('Individual ECDF Distribution', fontsize=22)



plt.show()