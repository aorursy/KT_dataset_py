## Load packages

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize

from scipy import stats

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

from ggplot import *
## Load data into Python



alldata = pd.read_csv('../input/creditcard.csv')
## What is in the data

print(alldata.columns)

print(alldata.shape)
alldata.head()
## Add dummy variable Source 

## And convert known class type variables 



alldata['Source']='Train'

alldata['Class']=alldata['Class'].astype(object)

alldata['ClassInt']=alldata['Class'].astype(int)

alldata['ClassYN']=['Y' if x == 1 else 'N' for x in alldata.Class]

alldata['LogAmt']=np.log(alldata.Amount+0.5)
## Numerical and Categorical data types

alldata_dtype=alldata.dtypes

display_nvar = len(alldata.columns)

alldata_dtype_dict = alldata_dtype.to_dict()

alldata.dtypes.value_counts()
def var_desc(dt):

    print('--------------------------------------------')

    for c in alldata.columns:

        if alldata[c].dtype==dt:

            t1 = alldata[alldata.Class==0][c]

            t2 = alldata[alldata.Class==1][c]

            if dt=="object":

                f1 = t1[pd.isnull(t1)==False].value_counts()

                f2 = t2[pd.isnull(t2)==False].value_counts()

            else:

                f1 = t1[pd.isnull(t1)==False].describe()

                f2 = t2[pd.isnull(t2)==False].describe()

            m1 = t1.isnull().value_counts()

            m2 = t2.isnull().value_counts()

            f = pd.concat([f1, f2], axis=1)

            m = pd.concat([m1, m2], axis=1)

            f.columns=['NoFraud','Fraud']

            m.columns=['NoFraud','Fraud']

            print(dt+' - '+c)

            print('UniqValue - ',len(t1.value_counts()),len(t2.value_counts()))

            print(f.sort_values(by='NoFraud',ascending=False))

            print()



            m_print=m[m.index==True]

            if len(m_print)>0:

                print('missing - '+c)

                print(m_print)

            else:

                print('NO Missing values - '+c)

            if dt!="object":

                if len(t1.value_counts())<=10:

                    c1 = t1.value_counts()

                    c2 = t2.value_counts()

                    c = pd.concat([c1, c2], axis=1)

                    f.columns=['NoFraud','Fraud']

                    print(c)

            print('--------------------------------------------')
var_desc('int64')
var_desc('float64')
# Top 10 correlated variables

corrmat = alldata.corr()

k = 8 #number of variables for heatmap

cols = corrmat.nlargest(k, 'ClassInt')['ClassInt'].index

cm = np.corrcoef(alldata[cols].values.T)

f, ax = plt.subplots(figsize=(8, 8))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
## Violin plot of the same numeric variables



dt = 'float64'

sel_col = alldata.columns[alldata.dtypes==dt]



plt.figure(figsize=(12,len(sel_col)*4))

gs = gridspec.GridSpec(len(sel_col),1)

for i, cn in enumerate(alldata[sel_col]):

    ax = plt.subplot(gs[i])

    data_1  = pd.concat([alldata[cn], alldata.ClassYN], axis=1)

    data_2  = pd.melt(data_1,id_vars=cn)

    sns.violinplot( x=cn, y='variable', hue="value"

                   ,data=data_2, palette="Set2", split=True

                   ,inner="quartile")

    ax.set_xlabel('')

    ax.set_title('Violin plot of : ' + str(cn))

plt.show()



    
## Treating skewed continuous data - transformation

## already did it
## Remove unnessary columns

list_col_rm = ['Amount','Class','Source','ClassYN']

list_col_keep = alldata.columns.difference(list_col_rm)

print(list_col_keep)
## normalize numeric variables

##

excl_cols = ['ClassInt']

alldata_dtype_dict = alldata.dtypes.to_dict()

for c in alldata.columns:

    if c in list_col_keep and c not in excl_cols and alldata_dtype_dict[c]!='object':

        print('----------------------')

        print(c , alldata_dtype_dict[c])

        alldata[c] = (alldata[c]-alldata[c].mean())/(alldata[c].std())

print()

print(alldata.head())
## Data Sampling, we do a 80/20 split on train/test.. 

## will talk about stacking later on. 

trainY = alldata[alldata.Class==1].sample(frac=0.8)

trainN = alldata[alldata.Class==0].sample(frac=0.8)

train = pd.concat([trainY, trainN], axis = 0)

test  = alldata.loc[~alldata.index.isin(train.index)]

print(train.shape)

print(test.shape)
## Save processed data

## so I can download into a new notebook to build models

import pickle

file_obj = open('./data.p', 'wb') 

pickle.dump([train, test, list_col_keep], file_obj) 

file_obj.close()