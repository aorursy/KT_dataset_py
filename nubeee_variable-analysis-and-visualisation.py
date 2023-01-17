# Import modules

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import seaborn as sns

sns.set(style="white")
# import dataset

# train = pd.read_csv('../input/train.csv', na_values = -1)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv', na_values = -1)

# train and test

X_train = train.iloc[:,2:]

X_test = test.iloc[:,1:]

y_train = train.iloc[:,1]
from string import ascii_letters

# Compute the correlation matrix

corr = train.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# replot without _calc_ columns

import re

# print(train.columns)

col_to_remove = [col for col in train.columns if re.search(".*_calc_.*",col)]

# print(col_to_remove)

train_slim = train.drop(col_to_remove, axis = 1)

train_slim2 = train_slim.drop('id',axis = 1)

# print(train_slim2.columns)
corr = train_slim2.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
def strong_corr(corr_matrix, abs_threshold = 0.55):

    df = pd.DataFrame(corr_matrix)

    

    # create a df using column names  

    df_master = pd.DataFrame()

    for column in df.columns:

        dicts = {'col': df.columns, 'col_2': np.repeat(column,df.shape[1]),'value': df.loc[:,column]}

        df_sub = pd.DataFrame(dicts)

        df_master = pd.concat([df_sub,df_master])

        df_master_sort = df_master.sort_values(['value'], ascending = False)

        df_master_sort_filter = df_master_sort.loc[(df_master_sort['value'] < 1) & (abs(df_master_sort['value']) >= abs_threshold), :]

        

    return df_master_sort_filter
top_corr_df = strong_corr(train_slim2.corr(), abs_threshold = 0.55)

print(top_corr_df)
# run pair plot for the variables which have relatively strong correlations between each other

# plt.figure()

# sns.pairplot(data= train_slim2.loc[:,['ps_ind_12_bin','ps_ind_14','ps_car_13','ps_car_12','ps_reg_03','ps_reg_01','target']],

#              hue="target", dropna=True,  palette = 'Set1')

# plt.savefig("pairplot_strongcorr.png")
#plot again

corr = train[list(set(top_corr_df.index))].corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
for col in train.columns[0:10]:   

    plt.figure()

    sns.countplot(x= col, data=train, hue = "target", palette={1: "k", 0: "m"} )
# plot lmplot between reg 01, 03

reg = sns.lmplot(x="ps_reg_01", y="ps_reg_03", hue="target",

               truncate=True, size=5, data=train_slim2)