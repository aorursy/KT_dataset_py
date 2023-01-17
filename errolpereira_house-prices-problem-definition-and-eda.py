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



# Baic Information about the data

print(f'Training Shape : {train_df.shape}')

print(f'Testing Shape : {test_df.shape}')

print(f'Sample Submission Shape : {sub_df.shape}')
# Description of numeric variables

train_df.describe()
# Data info

train_df.info()
# Getting the number of continuous and categorical variables

cat_cols = [x for x in train_df.columns if train_df[x].dtype == 'object']

cont_cols = [x for x in train_df.columns if train_df[x].dtype != 'object']



print(f'Total categorical variables in the dataset: {len(cat_cols)}')

print(f'Total continuous variables in the dataset: {len(cont_cols)}')
# Plotting the distribution of the continuous variables

fig, axs = plt.subplots(8, 5, figsize=(20, 15))

axs = axs.flatten()



for i, column in zip(range(len(cont_cols)), cont_cols):

    sns.distplot(train_df[column], ax=axs[i], kde=False)

    plt.tight_layout()
# Appending categorical columns and removing continous columns

cat_cols.append(x for x in ['MSSubClass', 'OverallQual', 'OverallCond'])

for x in ['MSSubClass', 'OverallQual', 'OverallCond']:

    cont_cols.remove(x)
# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



# Distribution plot to see clearly

axs[0].set_title('Distribution of YearBuilt in Training Data')

sns.distplot(train_df.YearBuilt, ax=axs[0], color=color_pal[3]);



# Stats

print("======Train=======")

print(f"Range of years: {min(train_df.YearBuilt)}-{max(train_df.YearBuilt)}")

print(f"Max Number of houses: {train_df.YearBuilt.value_counts(sort='DESC').iloc[0]}" +

     f"\nYear : {train_df.YearBuilt.value_counts(sort='DESC').index[0]}")



# Distribution plot to see clearly

plt.title('Distribution of YearBuilt in test data')

sns.distplot(test_df.YearBuilt, ax=axs[1], color=color_pal[4]);



# Stats

print("======Test=======")

print(f"Range of years: {min(test_df.YearBuilt)}-{max(test_df.YearBuilt)}")

print(f"Max Number of houses: {test_df.YearBuilt.value_counts(sort='DESC').iloc[0]}" +

     f"\nYear : {test_df.YearBuilt.value_counts(sort='DESC').index[0]}")
# Distribution plot to see clearly

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()

axs[0].set_title('Distribution of MSSubclass in Training Data')

sns.distplot(train_df.MSSubClass, ax=axs[0], color=color_pal[1]);



axs[1].set_title('Distribution of MSSubclass in Test Data')

sns.distplot(test_df.MSSubClass, ax=axs[1], color=color_pal[2]);



# Stats

print(f'Train Peak observed at {train_df.MSSubClass.value_counts(sort="DESC").index[0]}')

print(f'Test Peak observed at {test_df.MSSubClass.value_counts(sort="DESC").index[0]}')
# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of MSZoning in train data')

sns.countplot(train_df.MSZoning, palette='gray', ax=axs[0]);



axs[1].set_title('Count of MSZoning in test data')

sns.countplot(test_df.MSZoning, palette='gray', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



axs[0].set_title('Distribution of LotFrontage in Training Data')

sns.distplot(train_df.LotFrontage, ax=axs[0], color=color_pal[5]);



axs[1].set_title('Distribution of LotFrontage in Test Data')

sns.distplot(test_df.LotFrontage, ax=axs[1], color=color_pal[6]);



print(f"Max Train distibution of LotFrontage at: {train_df.LotFrontage.value_counts(sort='DESC').index[0]}")

print(f"Max Test distibution of LotFrontage at: {test_df.LotFrontage.value_counts(sort='DESC').index[0]}")
# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



axs[0].set_title('Boxplot of LotFrontage in Training Data')

sns.boxplot(train_df.LotFrontage, ax=axs[0], color=color_pal[0]);



axs[1].set_title('Boxplot of LotFrontage in Test Data')

sns.boxplot(test_df.LotFrontage, ax=axs[1], color=color_pal[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,6))

axs = axs.flatten()



axs[0].set_title('Distribution of LotArea in Training Data')

sns.distplot(train_df.LotArea, ax=axs[0], color=color_pal[7]);



axs[1].set_title('Distribution of LotArea in Test Data')

sns.distplot(test_df.LotArea, ax=axs[1], color=color_pal[8]);



print(f"Max Train distibution of LotArea at: {train_df.LotArea.value_counts(sort='DESC').index[0]}")

print(f"Max Test distibution of LotArea at: {test_df.LotArea.value_counts(sort='DESC').index[0]}")
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,6))

axs = axs.flatten()



axs[0].set_title('Boxplot of LotArea in Training Data')

sns.boxplot(train_df.LotArea, ax=axs[0], color=color_pal[2]);



axs[1].set_title('Boxplot of LotArea in Test Data')

sns.boxplot(test_df.LotArea, ax=axs[1], color=color_pal[3]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of Street Type in train data')

sns.countplot(train_df.Street, palette='Dark2', ax=axs[0]);



axs[1].set_title('Count of Street Type in test data')

sns.countplot(test_df.Street, palette='Dark2', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of Alley Type in train data')

sns.countplot(train_df.Alley.fillna('NA'), palette='Blues', ax=axs[0]);



axs[1].set_title('Count of Alley Type in test data')

sns.countplot(test_df.Alley.fillna('NA'), palette='Blues', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of LotShape Type in train data')

sns.countplot(train_df.LotShape.fillna('NA'), palette='Purples', ax=axs[0]);



axs[1].set_title('Count of LotShape Type in test data')

sns.countplot(test_df.LotShape.fillna('NA'), palette='Purples', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of LandContour Type in train data')

sns.countplot(train_df.LandContour.fillna('NA'), palette='Reds', ax=axs[0]);



axs[1].set_title('Count of LandContour Type in test data')

sns.countplot(test_df.LandContour.fillna('NA'), palette='Reds', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of Utilities Type in train data')

sns.countplot(train_df.Utilities.fillna('NA'), palette='Greens', ax=axs[0]);



axs[1].set_title('Count of Utilities Type in test data')

sns.countplot(test_df.Utilities.fillna('NA'), palette='Greens', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of LotConfig Type in train data')

sns.countplot(train_df.LotConfig.fillna('NA'), palette='Oranges', ax=axs[0]);



axs[1].set_title('Count of LotConfig Type in test data')

sns.countplot(test_df.LotConfig.fillna('NA'), palette='Oranges', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of LandSlope Type in train data')

sns.countplot(train_df.LandSlope.fillna('NA'), palette='Oranges', ax=axs[0]);



axs[1].set_title('Count of LandSlope Type in test data')

sns.countplot(test_df.LandSlope.fillna('NA'), palette='Oranges', ax=axs[1]);
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# Countplot

axs[0].set_title('Count of Neighborhood Type in train data')

sns.countplot(train_df.Neighborhood.fillna('NA'), palette='twilight_shifted', ax=axs[0]);

axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90, ha="right");



axs[1].set_title('Count of LandSlope Type in test data')

sns.countplot(test_df.Neighborhood.fillna('NA'), palette='twilight_shifted', ax=axs[1]);

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, ha="right");



print(f'Number of Neighborhoods(Train) : {len(np.unique(train_df.Neighborhood))}')

print(f'Number of Neighborhoods(Test) : {len(np.unique(test_df.Neighborhood))}')
# Figure

fig, axs = plt.subplots(2, 2, figsize=(16,10))

axs = axs.flatten()



# Condition1

axs[0].set_title('Count of Condition1 Type in train data')

sns.countplot(train_df.Condition1.fillna('NA'), palette='autumn', ax=axs[0]);

axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90, ha="right");



axs[1].set_title('Count of Condition1 Type in test data')

sns.countplot(test_df.Condition1.fillna('NA'), palette='autumn', ax=axs[1]);

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, ha="right");



# Condition2

axs[2].set_title('Count of Condition2 Type in train data')

sns.countplot(train_df.Condition2.fillna('NA'), palette='autumn', ax=axs[2]);

axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=90, ha="right");



axs[3].set_title('Count of Condition2 Type in test data')

sns.countplot(test_df.Condition2.fillna('NA'), palette='autumn', ax=axs[3]);

axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=90, ha="right");



plt.tight_layout()
# Figure

fig, axs = plt.subplots(1, 2, figsize=(16,5))

axs = axs.flatten()



# BedroomAbvGr

axs[0].set_title('Count of BedroomAbvGr Type in train data')

sns.countplot(train_df.BedroomAbvGr.fillna('NA'), palette='Blues', ax=axs[0]);



axs[1].set_title('Count of BedroomAbvGr Type in test data')

sns.countplot(test_df.BedroomAbvGr.fillna('NA'), palette='Blues', ax=axs[1]);

# Figure

fig, axs = plt.subplots(1, 2, figsize=(15,5))

axs = axs.flatten()



axs[0].set_title('Distribution of MiscVal in Training Data')

sns.distplot(train_df.MiscVal, ax=axs[0], color=color_pal[7], kde=False);



axs[1].set_title('Distribution of MiscVal in Test Data')

sns.distplot(test_df.MiscVal, ax=axs[1], color=color_pal[8], kde=False);



print(f"Max Train distibution of MiscVal at: {train_df.MiscVal.value_counts(sort='DESC').index[0]}")

print(f"Max Test distibution of MiscVal at: {test_df.MiscVal.value_counts(sort='DESC').index[0]}")