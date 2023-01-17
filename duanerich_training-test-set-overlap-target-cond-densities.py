%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
unique_counts = pd.DataFrame(train.nunique(),columns=['train_nunique'])

unique_counts = unique_counts.join(pd.DataFrame(test.nunique(),columns=['test_nunique']))
unique_counts['count_in_train_not_test'] = np.nan

unique_counts['count_in_test_not_train'] = np.nan

cols = set(train.columns.values) - {'target'}

for c in cols:

    train_set_c = set(train[c])

    test_set_c = set(test[c])

    unique_counts.loc[c, 'count_in_train_not_test'] = len(train_set_c - test_set_c)

    unique_counts.loc[c, 'count_in_test_not_train'] = len(test_set_c - train_set_c)

unique_counts['interesting'] = unique_counts['count_in_train_not_test'] != 0

unique_counts['interesting'] = unique_counts['interesting'] | unique_counts['count_in_test_not_train'] != 0

unique_counts[unique_counts['interesting']]
interesting_cols = list(unique_counts[unique_counts['interesting']].index[2:].values)

train['which'] = 'training'

test['which'] = 'testing'

cols_pick = interesting_cols + ['which','id']

both = train[cols_pick].append(test[cols_pick])

both = pd.melt(both,id_vars=['id','which'], value_vars=interesting_cols)

plt.figure(figsize=(20,10))

ax = sns.boxplot(x="variable", y="value", hue="which", data=both.sample(frac=.05), palette="Set3")
col_groups = {'categorical': ['ps_ind_08_bin','ps_calc_20_bin','ps_ind_12_bin','ps_ind_13_bin',

                 'ps_ind_10_bin','ps_calc_18_bin','ps_ind_09_bin','ps_calc_17_bin',

                 'ps_calc_15_bin','ps_ind_16_bin','ps_calc_19_bin','ps_ind_17_bin',

                 'ps_ind_18_bin','ps_ind_07_bin','ps_ind_11_bin','ps_ind_06_bin',

                 'ps_calc_16_bin','ps_car_06_cat','ps_car_01_cat','ps_car_04_cat',

                 'ps_car_09_cat','ps_car_11_cat','ps_ind_05_cat','ps_ind_04_cat',

                 'ps_car_08_cat','ps_car_05_cat','ps_ind_02_cat','ps_car_03_cat',

                 'ps_car_07_cat','ps_car_02_cat','ps_car_10_cat'],

 'regression': ['ps_calc_11','ps_ind_14','ps_calc_05','ps_car_11','ps_calc_03',

                'ps_car_12','ps_reg_01','ps_ind_03','ps_calc_01','ps_ind_15',

                'ps_calc_14','ps_reg_03','ps_car_13','ps_calc_13','ps_car_14',

                'ps_calc_04','ps_calc_10','ps_reg_02','ps_calc_06','ps_calc_08',

                'ps_calc_02','ps_calc_12','ps_car_15','ps_calc_07','ps_ind_01',

                'ps_calc_09']}



def cat_hist_cond(cat_col):

    #This plots the conditional (on the target) histogram for a given categorical column name

    train_temp = train[[cat_col] + ['target']]

    targ_counts = train_temp.groupby('target').count()

    counts = train_temp.groupby([cat_col] + ['target']).size().reset_index(name='counts')

    for i in range(2):

        counts.loc[counts['target']==i,'counts'] = counts.loc[counts['target']==i,'counts']/targ_counts.loc[i].values[0]

    if cat_col == 'ps_car_11_cat':

        plt.figure(figsize=(24,6))

    else:

        plt.figure(figsize=(14,6))

    ax = sns.barplot(x=cat_col,y='counts',hue='target',data=counts)

    plt.title('Normalized histogram of ' + cat_col + ' for each target outcome')

    plt.show()



def reg_dens_cond(reg_col,width):

    #This plots the conditional (on the target) density for a given regression column name

    train_temp = train[[reg_col] + ['target']]

    #We standardize so we can use the same KDE width for both distributions.

    train_temp.loc[:,reg_col] -= np.mean(train_temp.loc[:,reg_col])

    train_temp.loc[:,reg_col] /= np.std(train_temp.loc[:,reg_col])

    plt.figure(figsize=(14,6))

    sns.kdeplot(train_temp.loc[train_temp['target']==0,reg_col],bw=width,label="targ = 0")

    sns.kdeplot(train_temp.loc[train_temp['target']==1,reg_col],bw=width,label="targ = 1")

    plt.title('Density of ' + reg_col + ' for each target outcome')

    plt.show()



for cc in col_groups['categorical']:

    cat_hist_cond(cc)

for cc in col_groups['regression']:

    reg_dens_cond(cc,.2)