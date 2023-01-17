import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn import base

import scipy as sp

from scipy import stats

import seaborn as sns

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
test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

df = train

df.shape[0] - df.dropna().shape[0]

__seed = 0

__n_folds = 5

__nrows = None



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



from tqdm import tqdm_notebook



import numpy as np

import pandas as pd

pd.set_option('max_colwidth', 500)

pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)

from scipy.stats import chi2_contingency, kruskal, ks_2samp



from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import OneHotEncoder, StandardScaler



from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import StratifiedKFold, cross_validate



from string import ascii_lowercase

import random



# To avoid target leakage

folds1 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed)

folds2 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed+2)

folds3 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed+4)



def coef_vcramer(contingency_df):

    chi2 = chi2_contingency(contingency_df)[0]

    n = contingency_df.sum().sum()

    r, k = contingency_df.shape

    return np.sqrt(chi2 / (n * min((r-1), (k-1))))



def fit_describe_infos(train, test, __featToExcl = [], target_for_vcramer = None):

    '''Describe data and difference between train and test datasets.'''

    

    stats = []

    __featToAnalyze = [v for v in list(train.columns) if v not in __featToExcl]

    

    for col in tqdm_notebook(__featToAnalyze):

            

        dtrain = dict(train[col].value_counts())

        dtest = dict(test[col].value_counts())



        set_train_not_in_test = set(dtest.keys()) - set(dtrain.keys())

        set_test_not_in_train = set(dtrain.keys()) - set(dtest.keys())

        

        dict_train_not_in_test = {key:value for key, value in dtest.items() if key in set_train_not_in_test}

        dict_test_not_in_train = {key:value for key, value in dtrain.items() if key in set_test_not_in_train}

            

        nb_moda_test, nb_var_test = len(dtest), pd.Series(dtest).sum()

        nb_moda_abs, nb_var_abs = len(dict_train_not_in_test), pd.Series(dict_train_not_in_test).sum()

        nb_moda_train, nb_var_train = len(dtrain), pd.Series(dtrain).sum()

        nb_moda_abs_2, nb_var_abs_2 = len(dict_test_not_in_train), pd.Series(dict_test_not_in_train).sum()

        

        if not target_for_vcramer is None:

            vc = coef_vcramer(pd.crosstab(train[target_for_vcramer], train[col].fillna(-1)))       

        else:

            vc = 0

            

        stats.append((col, round(vc, 3), train[col].nunique()

            , str(nb_moda_abs) + '   (' + str(round(100 * nb_moda_abs / nb_moda_test, 1))+'%)'

            , str(nb_moda_abs_2) +'   (' + str(round(100 * nb_moda_abs_2 / nb_moda_train, 1))+'%)'

            , str(train[col].isnull().sum()) +'   (' + str(round(100 * train[col].isnull().sum() / train.shape[0], 1))+'%)'

            , str(test[col].isnull().sum()) +'   (' + str(round(100 * test[col].isnull().sum() / test.shape[0], 1))+'%)'

            , str(round(100 * train[col].value_counts(normalize = True, dropna = False).values[0], 1))

            , train[col].dtype))

            

    df_stats = pd.DataFrame(stats, columns=['Feature', "Target Cramer's V"

        , 'Unique values (train)', "Unique values in test not in train (and %)"

        , "Unique values in train not in test (and %)"

        , 'NaN in train (and %)', 'NaN in test (and %)', '% in the biggest cat. (train)'

        , 'dtype'])

    

    if target_for_vcramer is None:

        df_stats.drop("Target Cramer's V", axis=1, inplace=True)

            

    return df_stats, dict_train_not_in_test, dict_test_not_in_train



dfi, _, _ = fit_describe_infos(train, test, __featToExcl=['target'], target_for_vcramer='target')

dfi
summary = resumetable(train)

summary



total = len(train)

bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

#Looking the V's features

import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20)) # size of figure



# loop to get column and the count of plots

for n, col in enumerate(train[bin_cols]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=train, hue='target', palette='hls') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

    sizes=[] # Get highest values in y

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=14) 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

    

plt.show()
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','nom_7','nom_8']



def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):

    

    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart

    plt.figure(figsize=(17, 35)) # size of figure



    # loop to get column and the count of plots

    for n, col in enumerate(train[cols]): 

        tmp = pd.crosstab(train[col], train['target'], normalize='index') * 100

        tmp = tmp.reset_index()

        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



        ax = plt.subplot(grid[n]) # feeding the figure of grid

        sns.countplot(x=col, data=train, order=list(tmp[col].values) , color='green') 

        ax.set_ylabel('Count', fontsize=15) # y axis label

        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



        # twinX - to build a second yaxis

        gt = ax.twinx()

        gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

        gt.set_ylabel("Target %True(1)", fontsize=16)

        sizes=[] # Get highest values in y

        for p in ax.patches: # loop to all objects

            height = p.get_height()

            sizes.append(height)

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center", fontsize=14) 

        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





    plt.subplots_adjust(hspace = 0.5, wspace=.3)

    plt.show()



ploting_cat_fet(train, nom_cols, vis_row=5, vis_col=2)



test['target'] = 'test'

df = pd.concat([train, test], axis=0, sort=False )



print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')



train, df_test = df[df['target'] != 'test'], df[df['target'] == 'test'].drop('target', axis=1)

del df



train.head()



ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3']



def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):

    

    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart

    plt.figure(figsize=(17, 35)) # size of figure



    # loop to get column and the count of plots

    for n, col in enumerate(train[cols]): 

        tmp = pd.crosstab(train[col], train['target'], normalize='index') * 100

        tmp = tmp.reset_index()

        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



        ax = plt.subplot(grid[n]) # feeding the figure of grid

        sns.countplot(x=col, data=train, order=list(tmp[col].values) , color='green') 

        ax.set_ylabel('Count', fontsize=15) # y axis label

        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



        # twinX - to build a second yaxis

        gt = ax.twinx()

        gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

        gt.set_ylabel("Target %True(1)", fontsize=16)

        sizes=[] # Get highest values in y

        for p in ax.patches: # loop to all objects

            height = p.get_height()

            sizes.append(height)

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center", fontsize=14) 

        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





    plt.subplots_adjust(hspace = 0.5, wspace=.3)

    plt.show()



ploting_cat_fet(train, ord_cols, vis_row=5, vis_col=2)

train['ord_5_ot'] = 'Others'

train.loc[train['ord_5'].isin(train['ord_5'].value_counts()[:25].sort_index().index), 'ord_5_ot'] = train['ord_5']



tmp = pd.crosstab(train['ord_4'], train['target'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

plt.figure(figsize=(15,12))



plt.subplot(211)

ax = sns.countplot(x='ord_4', data=train, order=list(tmp['ord_4'].values) , color='green') 

ax.set_ylabel('Count', fontsize=17) # y axis label

ax.set_title('ord_4 Distribution with Target %ratio', fontsize=20) # title label

ax.set_xlabel('ord_4 values', fontsize=17) # x axis label

# twinX - to build a second yaxis

gt = ax.twinx()

gt = sns.pointplot(x='ord_4', y='Yes', data=tmp,

                   order=list(tmp['ord_4'].values),

                   color='black', legend=False)

gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

gt.set_ylabel("Target %True(1)", fontsize=16)



tmp = pd.crosstab(train['ord_5_ot'], train['target'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



plt.subplot(212)

ax1 = sns.countplot(x='ord_5_ot', data=train,

                   order=list(train['ord_5_ot'].value_counts().sort_index().index) ,

                   color='green') 

ax1.set_ylabel('Count', fontsize=17) # y axis label

ax1.set_title('TOP 25 ord_5 and "others" Distribution with Target %ratio', fontsize=20) # title label

ax1.set_xlabel('ord_5 values', fontsize=17) # x axis label

# twinX - to build a second yaxis

gt = ax1.twinx()

gt = sns.pointplot(x='ord_5_ot', y='Yes', data=tmp,

                   order=list(train['ord_5_ot'].value_counts().sort_index().index),

                   color='black', legend=False)

gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

gt.set_ylabel("Target %True(1)", fontsize=16)



plt.subplots_adjust(hspace = 0.4, wspace=.3)



plt.show()



ord_5_count = train['ord_5'].value_counts().reset_index()['ord_5'].values

plt.figure(figsize=(12,6))



g = sns.distplot(ord_5_count, bins= 50)

g.set_title("Frequency of ord_5 category values", fontsize=22)

g.set_xlabel("Total of entries in ord_5 category's", fontsize=18)

g.set_ylabel("Density", fontsize=18)



plt.show()
from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)



# Transforming ordinal Features

train.ord_1 = train.ord_1.astype(ord_1)

train.ord_2 = train.ord_2.astype(ord_2)

train.ord_3 = train.ord_3.astype(ord_3)

train.ord_4 = train.ord_4.astype(ord_4)



# test dataset

test.ord_1 = test.ord_1.astype(ord_1)

test.ord_2 = test.ord_2.astype(ord_2)

test.ord_3 = test.ord_3.astype(ord_3)

test.ord_4 = test.ord_4.astype(ord_4)



# Geting the codes of ordinal categoy's - train

train.ord_1 = train.ord_1.cat.codes

train.ord_2 = train.ord_2.cat.codes

train.ord_3 = train.ord_3.cat.codes

train.ord_4 = train.ord_4.cat.codes



# Geting the codes of ordinal categoy's - test

test.ord_1 = test.ord_1.cat.codes

test.ord_2 = test.ord_2.cat.codes

test.ord_3 = test.ord_3.cat.codes

test.ord_4 = test.ord_4.cat.codes

train[['ord_0', 'ord_1', 'ord_2', 'ord_3']].head(10)
date_cols = ['day', 'month']



# Calling the plot function with date columns

ploting_cat_fet(train, date_cols, vis_row=5, vis_col=2)
def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



train = date_cyc_enc(train, 'day', 7)

test = date_cyc_enc(test, 'day', 7) 



train = date_cyc_enc(train, 'month', 12)

test = date_cyc_enc(test, 'month', 12)
import string



# Then encode 'ord_5' using ACSII values



# Option 1: Add up the indices of two letters in string.ascii_letters

train['ord_5_oe_add'] = train['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

df_test['ord_5_oe_add'] = df_test['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))



# Option 2: Join the indices of two letters in string.ascii_letters

df_train['ord_5_oe_join'] = df_train['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

df_test['ord_5_oe_join'] = df_test['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))



# Option 3: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately

df_train['ord_5_oe1'] = df_train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

df_test['ord_5_oe1'] = df_test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))



df_train['ord_5_oe2'] = df_train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

df_test['ord_5_oe2'] = df_test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))



for col in ['ord_5_oe1', 'ord_5_oe2', 'ord_5_oe_add', 'ord_5_oe_join']:

    df_train[col]= df_train[col].astype('float64')

    df_test[col]= df_test[col].astype('float64')