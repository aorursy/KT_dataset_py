# Importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import cycle

import warnings



# Style

pd.set_option('max_columns', 50)

plt.style.use('bmh')

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



# Supressing Warnings

warnings.filterwarnings('ignore')
# Reading data

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Getting the number of continuous and categorical variables

cat_cols = [x for x in train_df.columns if train_df[x].dtype == 'object']

cont_cols = [x for x in train_df.columns if train_df[x].dtype != 'object']



# Appending categorical columns and removing continous columns

cat_cols.append(x for x in ['MSSubClass', 'OverallQual', 'OverallCond'])

for x in ['MSSubClass', 'OverallQual', 'OverallCond']:

    cont_cols.remove(x)
# Pairplot for continuous variables

sns.pairplot(train_df[cont_cols]);
# Scatter plot for train for yearbuilt

plt.figure(figsize=(10, 5))

plt.title('Distribution of YearBuilt vs Selling Price in Training Data')

sns.scatterplot(train_df.YearBuilt, train_df.SalePrice, color=color_pal[1]);
# LotArea Vs SalePrice

plt.figure(figsize=(10, 5))

plt.title('LotArea vs Selling Price')

sns.scatterplot(train_df.LotArea, train_df.SalePrice, color=color_pal[1]);
plt.figure(figsize=(10, 5))

plt.title('Log(LotArea) vs Selling Price')

sns.scatterplot(np.log1p(train_df.LotArea), train_df.SalePrice, color=color_pal[1]);
# SalePrice with and without alley

df_with_Alley = train_df.loc[((train_df.Alley == 'Grvl') | (train_df.Alley == 'Paved'))]

df_without_Alley = train_df.loc[train_df.Alley.isnull()]



# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



axs[0].set_title('SalePrice distribution with Alley')

sns.distplot(np.log1p(df_with_Alley.SalePrice), ax=axs[0], color=color_pal[2]);



axs[1].set_title('SalePrice distribution without Alley')

sns.distplot(np.log1p(df_without_Alley.SalePrice), ax=axs[1], color=color_pal[2]);



print(f"Mean SalePrice for houses with Alley : {df_with_Alley.SalePrice.mean()}")

print(f"Mean SalePrice for houses without Alley : {df_without_Alley.SalePrice.mean()}")
# Houses with regular shape and irregular shape

df_regular_LotShape = train_df.loc[(train_df.LotShape == 'Reg')]

df_irregular_LotShape = train_df.loc[(train_df.LotShape != 'Reg')]



# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



axs[0].set_title('SalePrice distribution with regular LotShape')

sns.distplot(np.log1p(df_regular_LotShape.SalePrice), ax=axs[0], color=color_pal[3]);



axs[1].set_title('SalePrice distribution irregular Lotshape')

sns.distplot(np.log1p(df_irregular_LotShape.SalePrice), ax=axs[1], color=color_pal[3]);



print(f"Mean SalePrice for houses with regular LotShape : {df_regular_LotShape.SalePrice.mean()}")

print(f"Mean SalePrice for houses with irregular Lotshape : {df_irregular_LotShape.SalePrice.mean()}")
# Houses with utilities and those with not

df_all_utilities = train_df.loc[(train_df.Utilities == 'AllPub')]

df_few_utility = train_df.loc[(train_df.Utilities != 'AllPub')]



# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



axs[0].set_title('SalePrice distribution with all Utilities')

sns.distplot(df_all_utilities.SalePrice, ax=axs[0], color=color_pal[4]);



axs[1].set_title('SalePrice distribution with few Utilities')

sns.distplot(df_few_utility.SalePrice, ax=axs[1], color=color_pal[4]);



print(f"Mean SalePrice for houses with all Utilities : {df_all_utilities.SalePrice.mean()}")

print(f"Mean SalePrice for houses with few Utilities : {df_few_utility.SalePrice.mean()}")
# Checking on the testing data. Amount of houses which have NA values in them.

test_df.Utilities.isnull().sum()
# Price Distribution based on each neighbourhood.

neighborhoods = [x for x in train_df.Neighborhood.unique()]



# Figure

fig, axs = plt.subplots(len(neighborhoods)//5 + len(neighborhoods)%5, 5, figsize=(30, 15), sharex=True)

axs = axs.flatten()



# Mean SalePrice

mean = []



for i, neighbor in enumerate(neighborhoods):

    axs[i].set_title(f'SalePrice distribution for {neighbor}')

    sns.distplot(train_df.loc[train_df.Neighborhood == neighbor]['SalePrice'], ax=axs[i], color=color_pal[5]);

    mean.append(train_df.loc[train_df.Neighborhood == neighbor]['SalePrice'].mean())

    print(f"Mean Sale Price for {neighbor}: {mean[i]}")

    

plt.tight_layout()

plt.show();



print(f'Max mean sale price is for neighborhood {neighborhoods[np.argmax(mean)]} : {max(mean)}')

print(f'Min mean sale price is for neighborhood {neighborhoods[np.argmin(mean)]} : {min(mean)}')
# Get only houses with MiscVal greater than 0

df_miscVal = train_df.loc[train_df.MiscVal > 0]



# MiscVal vs SalePrice

plt.title('SalePrice vs MiscVal')

sns.scatterplot(df_miscVal.SalePrice, df_miscVal.MiscVal, color=color_pal[5]);
# Creating a column to calculate the total floor area of the house.

train_df['Total_Floor_Area'] = train_df['1stFlrSF'] + train_df['2ndFlrSF']



# Total_Floor_Area vs SalePrice

plt.figure(figsize=(8,5))

plt.title('SalePrice vs Total_Floor_Area')

sns.scatterplot(train_df.SalePrice, train_df.Total_Floor_Area, color=color_pal[6]);
# Using the same df for irregular and regular LotShape

print(f"Mean Area of Regular LotShape Properties : {df_regular_LotShape.LotArea.mean()}")

print(f"Mean Area of Irregular LotShape Properties : {df_irregular_LotShape.LotArea.mean()}")
columns_plot = ['MSZoning', 'MSSubClass', 'Condition1', 'Condition2',

               'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',

               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

               'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation']



# Plotting catplotss for above columns

fig, ax = plt.subplots(4, 4, figsize=(25,20))

ax = ax.flatten()



for i, col in enumerate(columns_plot):

    ax[i].tick_params(axis='x', labelsize=10, rotation=90)

    ax[i].set_xlabel('label', fontsize=17, position=(.5,20))

    ax[i].set_ylabel('label', fontsize=17)

    ax[i] = sns.stripplot(x=col, y="SalePrice", data=train_df, ax=ax[i])

fig.suptitle('Categorical Features Vs SalePrice Overview', position=(.5,1.1), fontsize=20)

fig.tight_layout()



fig.show()