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
# first import all necessary libraries

import itertools

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# set seaborn's default settings

sns.set() 
df = pd.read_csv("../input/car-mpg/mpg_raw.csv")

df.head()
# so now the data is in rectangular form with 398 entries each having 9 distinct properties

df.shape
# let's list all the columns

columns = list(df.columns)

columns
# we now describe the properties of this dataframe like column datatype etc.

df.info()
cats = list(df.select_dtypes(include=['object']).columns)

nums = list(df.select_dtypes(exclude=['object']).columns)

print(f'categorical variables:  {cats}')

print(f'numerical variables:  {nums}')
# let's inspect how many unique values are there in each column.

df.nunique(axis=0)
# cylinders and model_year also seems to be categorical so lets update the lists

cats.extend(['cylinders', 'model_year'])

nums.remove('cylinders')

nums.remove('model_year')



print(f'categorical variables:  {cats}')

print(f'numerical variables:  {nums}')
# check for `nans` in each column

df.isna().sum()
# let's print these 6 `nan` containing rows 

df[df.isnull().any(axis=1)]
# nan rows proportion in data

6 / len(df)
# for now remove all nan rows as they are just 1.5%

df = df[~df.isnull().any(axis=1)]

df.reset_index(inplace=True)

df.drop('index', inplace=True, axis=1)

df.shape
# find total duplicate entries and drop them if any

print(f'total duplicate rows: {df.duplicated().sum()}')



# drop duplicate rows if any

df = df[~df.duplicated()]

df.shape
# before we move ahead it's a good practice to group all variables together having same type.

df = pd.concat((df[cats], df[nums]), axis=1)

df.head()
num_rows, num_cols = df.shape
# save this cleaned df to csv

df.to_csv('mpg_cleaned.csv', index=False)
# let's import the cleaned version of mpg although no need here because we already updated df

df = pd.read_csv("mpg_cleaned.csv")
print(f'categorical variables:  {cats}')
df_cat = df.loc[:, 'origin':'model_year']

df_cat.head()
# remove extra spaces if any

for col in ['origin', 'name']:

    df_cat[col] = df_cat[col].apply(lambda x: ' '.join(x.split()))
df_cat['mpg_level'] = df['mpg'].apply(lambda x: 'low' if x<17 else 'high' if x>29 else 'medium')

cats.append('mpg_level')

print(f'categorical variables:  {cats}')
# let's look at the unique categories in `origin`, `cylinders` & `model_year`

# we are leaving `name` because it is almost unique for each entry (nothing interesting)

print(f"categories in origin: {pd.unique(df_cat['origin'])}")

print(f"categories in cylinders: {pd.unique(df_cat['cylinders'])}")

print(f"categories in model_year: {pd.unique(df_cat['model_year'])}")
# Although descriptive stats for categorical variables are not much informatic but still it's worth looking once.

# Also pandas describe function is only for numeric data and in df_cat `cylinders` & `model_year` are the only numeric type.

df_cat.describe()
fig = plt.figure(1, (14, 8))



for i,cat in enumerate(df_cat.drop(['name'], axis=1).columns):

    ax = plt.subplot(2,2,i+1)

    sns.countplot(df_cat[cat], order=df_cat[cat].value_counts().index)

    ax.set_xlabel(None)

    ax.set_title(f'Distribution of {cat}')

    plt.tight_layout()



plt.show()
# calculate proportion of dominant classes in each category

for i,cat in enumerate(df_cat.drop(['name'], axis=1).columns):

    val_counts = df_cat[cat].value_counts()

    dominant_frac = val_counts.iloc[0] / num_rows

    print(f'`{val_counts.index[0]}` alone contributes to {round(dominant_frac * 100, 2)}% of {cat}')
# count of different cylinders

df_cat.cylinders.value_counts()
print(f'total unique categories in `name`: {df_cat.name.nunique()}')

print(f"\nunique categories in `name`:\n\n {df_cat.name.unique()}")
# extract car company from `name`

df_cat['car_company'] = df_cat['name'].apply(lambda x: x.split()[0])



# remove car company from `name` and rename to `car_name`

df_cat['car_name'] = df_cat['name'].apply(lambda x: ' '.join(x.split()[1:]))

df_cat.drop('name', axis=1, inplace=True)



cats.extend(['car_company', 'car_name'])

cats.remove('name')



print(f'categorical variables:  {cats}')

df_cat.head()
# now check for total unique values in `car_company`

print(f'total unique categories in `car_company`: {df_cat.car_company.nunique()}')

print(f"\nunique categories in `car_company`:\n\n {df_cat.car_company.unique()}")
fig = plt.figure(1, (18, 4))



ax1 = plt.subplot(1,1,1)

sns.countplot(df_cat['car_company'], order=df_cat['car_company'].value_counts().index)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=75)



plt.show()
df_cat.car_company.value_counts()[:2]
combos = itertools.combinations(['origin', 'cylinders', 'mpg_level'], 2)



fig = plt.figure(1, (18, 8))



i = 0

for pair in combos:

#     i+=1

#     ax = plt.subplot(2,3,i)

#     sns.countplot(x=pair[0], hue=pair[1], data=df_cat)

#     ax.set_xlabel(None)

#     ax.set_title(f'{pair[0]} bifurcated by {pair[1]}')

#     plt.tight_layout()



    i+=1

    ax = plt.subplot(2,3,i)

    sns.countplot(x=pair[1], hue=pair[0], data=df_cat)

    ax.set_xlabel(None)

    ax.set_title(f'{pair[1]} bifurcated by {pair[0]}')

    plt.tight_layout()
sns.catplot(x='mpg_level', hue='cylinders', col='origin', data=df_cat, kind='count')

plt.show()
fig = plt.figure(1, (18,4))

sns.countplot(x='model_year', hue='mpg_level', data=df_cat)

sns.relplot(x='model_year', y='mpg', data=df)

plt.show()
fig = plt.figure(1, (18,4))

sns.countplot(x='model_year', hue='cylinders', data=df_cat)

plt.show()
fig = plt.figure(1, (18,4))

sns.countplot(x='model_year', hue='origin', data=df_cat)

plt.show()
top_car_companies = df_cat.car_company.value_counts()[:15].index

top_car_companies
df_cat_top_comp = df_cat[df_cat.car_company.isin(top_car_companies)]

df_cat_top_comp.shape
fig = plt.figure(1, (18,12))



for i,cat in enumerate(['mpg_level', 'origin', 'cylinders']):

    ax = plt.subplot(3,1,i+1)

    sns.countplot(x='car_company', hue=cat, data=df_cat_top_comp)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)

    plt.tight_layout()
df = pd.concat((df_cat.loc[:, 'origin':'car_company'], df.loc[:, 'mpg':'acceleration']), axis=1)

df.head()
# save these changes to new file

df.to_csv("mpg_cated.csv", index=False)
df = pd.read_csv("mpg_cated.csv")

df.head()
print(f'numerical variables:  {nums}')
df_num = df.loc[:, 'mpg':]
df_num.describe()
rows = len(nums)

cols = 3



fig = plt.figure(1, (18, rows*3))



i = 0

for col in nums:

    

    i += 1

    ax1 = plt.subplot(rows, cols,i)

#     ax1.hist(df[col], alpha=0.6)

    sns.distplot(df_num[col])

    ax1.set_xlabel(None)

    ax1.set_title(f'{col} distribution')

    plt.tight_layout()



    i += 1

    ax2 = plt.subplot(rows, cols,i)

    sns.violinplot(df_num[col])

    sns.swarmplot(df_num[col], alpha=0.6, color='k')

    ax2.set_xlabel(None)

    ax2.set_title(f'{col} swarm-violin plot')

    plt.tight_layout()



    i += 1

    ax3 = plt.subplot(rows, cols,i)

    sns.boxplot(df_num[col], orient='h', linewidth=2.5)

    ax3.set_xlabel(None)

    ax3.set_title(f'{col} box plot')

    plt.tight_layout()
def tukey_outliers(x):

    q1 = np.percentile(x,25)

    q3 = np.percentile(x,75)

    

    iqr = q3-q1

    

    min_range = q1 - iqr*1.5

    max_range = q3 + iqr*1.5

    

    outliers = x[(x<min_range) | (x>max_range)]

    return outliers
for col in nums:

    outliers = tukey_outliers(df_num[col])

    if len(outliers):

        print(f"* {col} has these tukey outliers,\n{outliers}\n")

    else:

        print(f"* {col} doesn't have any tukey outliers.\n")
df.iloc[list(tukey_outliers(df_num.acceleration).index)]
df.iloc[list(tukey_outliers(df_num.horsepower).index)]
# see data is not scaled properly, we need to scale it for modelling but works fine for analysis.

fig = plt.figure(1, (12, 4))

ax = plt.subplot(1,1,1)

sns.boxplot(x="variable", y="value", data=pd.melt(df_num))

plt.xlabel(None)

plt.ylabel(None)

plt.show()
sns.pairplot(data=df, vars=nums, diag_kind='kde', hue='origin',

            plot_kws=dict(s=20, edgecolor="k", linewidth=0.1, alpha=0.5), diag_kws=dict(shade=True))

plt.show()
sns.heatmap(df_num.corr(method='spearman'), annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')

plt.show()
'''In this plot we analyze the relationship of horsepower & acceleration

bifurcated by origin, mpg_level & cylinders in a single plot.'''



sns.relplot(x='horsepower', y='acceleration', hue='mpg_level', #style='mpg_level',

            size='cylinders', col='origin', data=df, kind='scatter', sizes=(5, 100), alpha=0.6)

plt.show()
'''In this plot we analyze the relationship of weight & horsepower

bifurcated by origin, mpg_level & cylinders in a single plot.'''



sns.relplot(x='weight', y='horsepower', hue='mpg_level', #style='mpg_level',

            size='cylinders', col='origin', data=df, kind='scatter', sizes=(5, 100), alpha=0.6)

plt.show()
print('variation of numerical features with origin')



fig = plt.figure(1, (18, 8))



for idx,col in enumerate(nums):

    ax = plt.subplot(2, 3, idx+1)

    sns.boxenplot(x='origin', y=col, data=df)

    ax.set_xlabel(None)

    plt.tight_layout()
print('variation of numerical features with mpg_level')



fig = plt.figure(1, (18, 8))



for idx,col in enumerate(nums):

    ax = plt.subplot(2, 3, idx+1)

    sns.boxenplot(x='mpg_level', y=col, data=df)

    ax.set_xlabel(None)

    plt.tight_layout()
print('variation of numerical features with cylinders')



fig = plt.figure(1, (18, 14))



for idx,col in enumerate(nums):

    ax = plt.subplot(3, 2, idx+1)

    sns.boxenplot(x='cylinders', y=col, data=df)

    ax.set_xlabel(None)

    plt.tight_layout()
print('variation of numerical features with model_year')



fig = plt.figure(1, (18, 14))

# fig.tight_layout()



for idx,col in enumerate(nums):

    ax = plt.subplot(3, 2, idx+1)

    sns.violinplot(x='model_year', y=col, data=df)

    ax.set_xlabel(None)

    plt.tight_layout()
print('variation of numerical features with model_year bifurcated by origin')



fig = plt.figure(1, (18, 14))

# fig.tight_layout()



for idx,col in enumerate(nums):

    ax = plt.subplot(3, 2, idx+1)

    sns.lineplot(x="model_year", y=col, hue='origin', data=df, err_style='bars')

    ax.set_title(f'change in {col} as year progresses in different regions')

    ax.set_xlabel(None)

    plt.tight_layout()



plt.show()