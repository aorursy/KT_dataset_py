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



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)
## Load data into Python



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.columns)

print(train.shape)

print(test.shape)
train.head()
## Describe the output field

print(train['price_doc'].describe())

sns.distplot(train['price_doc'])
train['LogAmt']=np.log(train.price_doc+0.5)

print(train['LogAmt'].describe())

sns.distplot(train['LogAmt'])
## Merge data into one dataset to prepare compare between train and test

train_1 = train.copy()

train_1['Source']='Train'

test_1 = test.copy()

test_1['Source']='Test'

alldata = pd.concat([train_1, test_1])

print(alldata.shape)
## Numerical and Categorical data types

alldata_dtype=alldata.dtypes

display_nvar = len(alldata.columns)

alldata_dtype_dict = alldata_dtype.to_dict()

alldata.dtypes.value_counts()
def var_desc(dt):

    print('--------------------------------------------')

    for c in alldata.columns:

        if alldata[c].dtype==dt:

            t1 = alldata[alldata.Source=='Train'][c]

            t2 = alldata[alldata.Source=='Test'][c]

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

            f.columns=['Train','Test']

            m.columns=['Train','Test']

            print(dt+' - '+c)

            print('UniqValue - ',len(t1.value_counts()),len(t2.value_counts()))

            print(f.sort_values(by='Train',ascending=False))

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

                    f.columns=['Train','Test']

                    print(c)

            print('--------------------------------------------')
var_desc('int64')
var_desc('float64')
var_desc('object')
## Correlation

corrmat = train.ix[:,41:67].corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, vmax=.8, square=True,xticklabels=True,yticklabels=True,cbar=False,annot=True);
## Correlation

corrmat = train.corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, vmax=.8, square=True,xticklabels=False,yticklabels=False,cbar=False);
# Top 20 correlated variables

corrmat = train.corr()

k = 20 #number of variables for heatmap

cols = corrmat.nlargest(k, 'price_doc')['price_doc'].index

cm = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize=(12, 12))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X):

    '''X - pandas dataframe'''

    thresh = 5.0

    variables = range(X.shape[1])

 

    for i in np.arange(0, len(variables)):

        vif = [variance_inflation_factor(X.as_matrix(), ix) for ix in range(X.shape[1])]        

        print(vif)

        maxloc = vif.index(max(vif))

        if max(vif) > thresh:

            print('dropping ' + X[variables].columns[maxloc] + ' at index: ' + str(maxloc))

            del variables[maxloc]

 

    print('Remaining variables:')

    print(X.columns[variables])

    return X
col_sel = alldata.columns[alldata_dtype=='int64'].append(alldata.columns[alldata_dtype=='float64'])

X = train[col_sel].dropna()

print(X.shape)



calculate_vif_(X)
dt = 'object'

sel_col = train.columns[train.dtypes==dt]

sel_col = [t for t in sel_col if t not in ['Id','timestamp','LogAmt']] 



for sc in sel_col:

    data_1  = pd.concat([train[sc], train.LogAmt], axis=1)

    data_2  = pd.melt(data_1,id_vars='LogAmt')

    data_2  = data_2[pd.isnull(data_2.value)==False]

    p = ggplot(data_2, aes(x='value',y='LogAmt')) + geom_boxplot(alpha=0.5,) + theme(plot_margin = dict(right = 1, top=0.5), axis_title_x=sc)

    print(p)
dt = 'int64'

sel_col = train.columns[train.dtypes==dt]

sel_col = [t for t in sel_col if t not in ['Id','timestamp','LogAmt']] 



for sc in sel_col:

    data_1  = pd.concat([train[sc], train.LogAmt], axis=1)

    data_2  = pd.melt(data_1,id_vars='LogAmt')

    data_2  = data_2[pd.isnull(data_2.value)==False]

    p = ggplot(data_2, aes(x='value',y='LogAmt')) + geom_point(alpha=0.5) + theme(plot_margin = dict(right = 1, top=0.5), axis_title_x=sc)

    print(p)