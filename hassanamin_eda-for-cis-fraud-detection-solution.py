# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

#import plotly.plotly as py

#import plotly.graph_objs as go

#import plotly.tools as tls

#from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

#import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

#init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



# Preprocessing, modelling and evaluating

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_id = pd.read_csv("../input/train_identity.csv")

df_trans = pd.read_csv("../input/train_transaction.csv")

df_test = pd.read_csv("../input/test_transaction.csv")
df_trans.head(10)
df_trans.describe().transpose()
df_id.head(10)
df_id.describe().transpose()
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

#resumetable(df_trans)[:25]



summary = resumetable(df_trans)
summary.sort_values(by=['Missing'],ascending=False)
summary.sort_values(by=['Uniques'],ascending=False)
# Get columns with NA values percentage greater than given threshold(th)

def get_na(df,th):

    missing_values_count = df.isnull().sum()

    missing_values_percentage = (missing_values_count/df.shape[0])*100

    string_data = df #df.select_dtypes(include=["object"])

    missing_columns = [col for col in string_data if (string_data[col].isnull().sum()/df.shape[0])*100>th]



        

    return missing_columns,missing_values_percentage

# Drop selected columns in a given dataset

def drop_na(df,sel_col):

    good_cols = list(df.columns)

    for col in df.columns:

        if col in sel_col:

            good_cols.remove(col)

    return df[good_cols]



missing_columns, missing_values_percentage = get_na(df_trans,50)





print ('Missing Value Percentages :- \n',missing_values_percentage[missing_columns].sort_values(ascending=False))
df_new =drop_na(df_trans,missing_columns)



missing_columns, missing_values_percentage = get_na(df_new,50)

print ('Missing Value Percentages :- \n',missing_values_percentage[missing_columns])
#display_na(df_id)

missing_columns, missing_values_percentage = get_na(df_id,50)

print ('Missing Value Percentages :- \n',missing_values_percentage[missing_columns])

#display_na(df_test)

missing_columns, missing_values_percentage = get_na(df_test,50)

print ('Missing Value Percentages :- \n',missing_values_percentage[missing_columns])

total_cells = np.product(df_trans.shape)

total_missing = missing_values_count.sum()

print ("% of missing data = ",(total_missing/total_cells) * 100)
df_trans.describe()
import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

df_trans['TransactionDT'].value_counts().head(10)
df_trans['TransactionDT'].plot(kind='hist',figsize=(15, 5),label='train',bins=50, title='Train vs Test TransactionDT distribution')

df_test['TransactionDT'].plot(kind='hist',label='test',bins=50)

plt.legend()

plt.show()
df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)

total = len(df_trans)

tmp = pd.crosstab(df_trans['card4'], df_trans['isFraud'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)



plt.figure(figsize=(14,10))



# Card4 Distribution

plt.suptitle('Card 4 Distributions', fontsize=22)



plt.subplot(221)

g = sns.countplot(x='card4', data=df_trans)

# plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])

g.set_title("Card4 Distribution", fontsize=19)

g.set_ylim(0,420000)

g.set_xlabel("Card4 Category Names", fontsize=17)

g.set_ylabel("Count", fontsize=17)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center",fontsize=14) 







# Card4 by Target(isFraud)

plt.subplot(222)

g1 = sns.countplot(x='card4', hue='isFraud', data=df_trans)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])

gt = g1.twinx()

gt = sns.pointplot(x='card4', y='Fraud', data=tmp, 

                   color='black', legend=False, 

                   order=['discover', 'mastercard', 'visa', 'american express'])

gt.set_ylabel("% of Fraud Transactions", fontsize=16)

g1.set_title("Card4 by Target(isFraud)", fontsize=19)

g1.set_xlabel("Card4 Category Names", fontsize=17)

g1.set_ylabel("Count", fontsize=17)





# Card 4 Distribuition by ProductCD and Target

plt.subplot(212)

g3 = sns.boxenplot(x='card4', y='TransactionAmt', hue='isFraud', 

              data=df_trans[df_trans['TransactionAmt'] <= 2000] )

g3.set_title("Card 4 Distribuition by ProductCD and Target", fontsize=20)

g3.set_xlabel("Card4 Category Names", fontsize=17)

g3.set_ylabel("Transaction Values", fontsize=17)



plt.subplots_adjust(hspace = 0.6, top = 0.85)



plt.show()
cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']

for i in cards:

    print ("Unique ",i, " = ",df_trans[i].nunique())
fig, ax = plt.subplots(1, 4, figsize=(25,5))



sns.countplot(x="card4", ax=ax[0], data=df_trans.loc[df_trans['isFraud'] == 0])

ax[0].set_title('card4 isFraud=0', fontsize=14)

sns.countplot(x="card4", ax=ax[1], data=df_trans.loc[df_trans['isFraud'] == 1])

ax[1].set_title('card4 isFraud=1', fontsize=14)

sns.countplot(x="card6", ax=ax[2], data=df_trans.loc[df_trans['isFraud'] == 0])

ax[2].set_title('card6 isFraud=0', fontsize=14)

sns.countplot(x="card6", ax=ax[3], data=df_trans.loc[df_trans['isFraud'] == 1])

ax[3].set_title('card6 isFraud=1', fontsize=14)

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
print(pd.concat([df_trans[df_trans['isFraud'] == 1]['TransactionAmt']\

                 .quantile([.01, .1, .25, .5, .75, .9, .99])\

                 .reset_index(), 

                 df_trans[df_trans['isFraud'] == 0]['TransactionAmt']\

                 .quantile([.01, .1, .25, .5, .75, .9, .99])\

                 .reset_index()],

                axis=1, keys=['Fraud', "No Fraud"]))
total = len(df_trans)

tmp = pd.crosstab(df_trans['ProductCD'], df_trans['isFraud'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)



plt.figure(figsize=(14,10))

plt.suptitle('ProductCD Distributions', fontsize=22)



plt.subplot(221)

g = sns.countplot(x='ProductCD', data=df_trans)

# plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])



g.set_title("ProductCD Distribution", fontsize=19)

g.set_xlabel("ProductCD Name", fontsize=17)

g.set_ylabel("Count", fontsize=17)

g.set_ylim(0,500000)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 



plt.subplot(222)

g1 = sns.countplot(x='ProductCD', hue='isFraud', data=df_trans)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])

gt = g1.twinx()

gt = sns.pointplot(x='ProductCD', y='Fraud', data=tmp, color='black', order=['W', 'H',"C", "S", "R"], legend=False)

gt.set_ylabel("% of Fraud Transactions", fontsize=16)



g1.set_title("Product CD by Target(isFraud)", fontsize=19)

g1.set_xlabel("ProductCD Name", fontsize=17)

g1.set_ylabel("Count", fontsize=17)



plt.subplot(212)

g3 = sns.boxenplot(x='ProductCD', y='TransactionAmt', hue='isFraud', 

              data=df_trans[df_trans['TransactionAmt'] <= 2000] )

g3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=20)

g3.set_xlabel("ProductCD Name", fontsize=17)

g3.set_ylabel("Transaction Values", fontsize=17)



plt.subplots_adjust(hspace = 0.6, top = 0.85)



plt.show()
df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)

total = len(df_trans)

total_amt = df_trans.groupby(['isFraud'])['TransactionAmt'].sum().sum()

plt.figure(figsize=(16,6))



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

for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:

    df_trans[col] = df_trans[col].fillna("Miss")

    

def ploting_dist_ratio(df, col, lim=2000):

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)



    plt.figure(figsize=(20,5))

    plt.suptitle(f'{col} Distributions ', fontsize=22)



    plt.subplot(121)

    g = sns.countplot(x=col, data=df, order=list(tmp[col].values))

    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])

    g.set_title(f"{col} Distribution\nCound and %Fraud by each category", fontsize=18)

    g.set_ylim(0,400000)

    gt = g.twinx()

    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),

                       color='black', legend=False, )

    gt.set_ylim(0,20)

    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g.set_xlabel(f"{col} Category Names", fontsize=16)

    g.set_ylabel("Count", fontsize=17)

    for p in gt.patches:

        height = p.get_height()

        gt.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center",fontsize=14) 

        

    perc_amt = (df_trans.groupby(['isFraud',col])['TransactionAmt'].sum() / total_amt * 100).unstack('isFraud')

    perc_amt = perc_amt.reset_index()

    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)



    plt.subplot(122)

    g1 = sns.boxplot(x=col, y='TransactionAmt', hue='isFraud', 

                     data=df[df['TransactionAmt'] <= lim], order=list(tmp[col].values))

    g1t = g1.twinx()

    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, order=list(tmp[col].values),

                       color='black', legend=False, )

    g1t.set_ylim(0,5)

    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)

    g1.set_title(f"{col} by Transactions dist", fontsize=18)

    g1.set_xlabel(f"{col} Category Names", fontsize=16)

    g1.set_ylabel("Transaction Amount(U$)", fontsize=16)

        

    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)

    

    plt.show()
for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:

    ploting_dist_ratio(df_trans, col, lim=2500)
print("Card Features Quantiles: ")

print(df_trans[['addr1', 'addr2']].quantile([0.01, .025, .1, .25, .5, .75, .90,.975, .99]))
df_trans.loc[df_trans.addr1.isin(df_trans.addr1.value_counts()[df_trans.addr1.value_counts() <= 5000 ].index), 'addr1'] = "Others"

df_trans.loc[df_trans.addr2.isin(df_trans.addr2.value_counts()[df_trans.addr2.value_counts() <= 50 ].index), 'addr2'] = "Others"
def ploting_cnt_amt(df, col, lim=2000):

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    

    plt.figure(figsize=(16,14))    

    plt.suptitle(f'{col} Distributions ', fontsize=24)

    

    plt.subplot(211)

    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))

    gt = g.twinx()

    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),

                       color='black', legend=False, )

    gt.set_ylim(0,tmp['Fraud'].max()*1.1)

    gt.set_ylabel("%Fraud Transactions", fontsize=16)

    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)

    g.set_xlabel(f"{col} Category Names", fontsize=16)

    g.set_ylabel("Count", fontsize=17)

    g.set_xticklabels(g.get_xticklabels(),rotation=45)

    sizes = []

    for p in g.patches:

        height = p.get_height()

        sizes.append(height)

        g.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center",fontsize=12) 

        

    g.set_ylim(0,max(sizes)*1.15)

    

    #########################################################################

    perc_amt = (df_trans.groupby(['isFraud',col])['TransactionAmt'].sum() \

                / df_trans.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')

    perc_amt = perc_amt.reset_index()

    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    amt = df_trans.groupby([col])['TransactionAmt'].sum().reset_index()

    perc_amt = perc_amt.fillna(0)

    plt.subplot(212)

    g1 = sns.barplot(x=col, y='TransactionAmt', 

                       data=amt, 

                       order=list(tmp[col].values))

    g1t = g1.twinx()

    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 

                        order=list(tmp[col].values),

                       color='black', legend=False, )

    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)

    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)

    g.set_xticklabels(g.get_xticklabels(),rotation=45)

    g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)

    g1.set_xlabel(f"{col} Category Names", fontsize=16)

    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)

    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    

    

    for p in g1.patches:

        height = p.get_height()

        g1.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total_amt*100),

                ha="center",fontsize=12) 

        

    plt.subplots_adjust(hspace=.4, top = 0.9)

    plt.show()

    

ploting_cnt_amt(df_trans, 'addr1')
ploting_cnt_amt(df_trans, 'addr2')
resumetable(df_trans[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',

                      'C9', 'C10', 'C11', 'C12', 'C13', 'C14']])
df_trans[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',

                      'C9', 'C10', 'C11', 'C12', 'C13', 'C14']].describe()
df_trans.loc[df_trans.C1.isin(df_trans.C1\

                              .value_counts()[df_trans.C1.value_counts() <= 400 ]\

                              .index), 'C1'] = "Others"
ploting_cnt_amt(df_trans, 'C1')
df_trans.loc[df_trans.C2.isin(df_trans.C2\

                              .value_counts()[df_trans.C2.value_counts() <= 350 ]\

                              .index), 'C2'] = "Others"
ploting_cnt_amt(df_trans, 'C2')
df_trans['_Days'] = round((df_trans['TransactionDT'] / 3600) / 24, 0).astype(int)

df_trans['_Weekdays'] = df_trans['_Days'] % 7

df_trans['_Hours'] = (df_trans['TransactionDT'] / 3600).astype(int) % 24
df_trans['_Days'].max()
tmp = (df_trans.groupby('_Days')['TransactionAmt'].sum()).reset_index().sort_values('TransactionAmt', ascending=False)[:15]



plt.figure(figsize=(14,6))



g = sns.barplot(x='_Days', y='TransactionAmt', data=tmp)

g.set_title("TOP 15 Days with Highest Transaction Amounts", fontsize=20)

g.set_xlabel("Days with highest Transaction Total Amount", fontsize=17)

g.set_ylabel("Transaction Total Amount", fontsize=17)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total_amt*100),

           ha='center')



plt.show()
ploting_cnt_amt(df_trans, '_Weekdays')
ploting_cnt_amt(df_trans, '_Hours')