# import libs



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

# load data, slice american universities



cwu_df = pd.read_csv('/kaggle/input/cwur_2019.csv')

cwu_df = cwu_df.replace('-',np.nan) # clean '-' entries from html table

all_type = 'float64'

cwu_df = cwu_df.astype({'national_rank': all_type,

               'world_rank': all_type,

               'education_quality': all_type,

               'alumni_employment': all_type,

               'research_performance': all_type,

               'faculty_quality': all_type,

               'score': all_type}) # cast data types

# distribution for country representation on list

n=20

location_counts = cwu_df['location']

location_df = location_counts.value_counts(sort=True).to_frame('counts')

topn_df = location_df.head(n)

print('Top {} countries:\n{}'.format(n, topn_df))



y_pos = np.arange(len(topn_df))

dims = (12,9)

fig, ax = plt.subplots(figsize=dims)

ax.bar(y_pos,topn_df['counts'])

plt.xticks(y_pos, topn_df.index,rotation=60)

plt.title("Top {} Countries by Occurrence".format(n))

plt.xlabel("Country")

plt.ylabel("Counts")

plt.show()
all_data_na = (cwu_df.isna().sum() / len(cwu_df)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
# correlation matrix: ignore score

temp_df = cwu_df.drop(['score'],axis=1) #

corr_df = temp_df.dropna()

print("Number of entries after dropping NaN's: {}".format(len(corr_df)))

corrmat = corr_df.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat,

            cbar=True,

            annot=True,

            square = True,

            fmt='.2f',

            annot_kws={'size':10})

plt.show()
# correlation matrix: ignore score

temp_df = cwu_df.drop(['score','faculty_quality','education_quality'],axis=1) #

corr_df = temp_df.dropna()

print("Number of entries after dropping NaN's: {}".format(len(corr_df)))

corrmat = corr_df.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat,

            cbar=True,

            annot=True,

            square = True,

            fmt='.2f',

            annot_kws={'size':10})

plt.show()
# correlation matrix: ignore score

us_df = cwu_df.loc[cwu_df['location']=='USA'].drop('score',axis=1) #

corr_df = us_df.dropna()

print("Number of entries after dropping NaN's: {} out of {}".format(len(corr_df),len(us_df)))

corrmat = corr_df.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat,

            cbar=True,

            annot=True,

            square = True,

            fmt='.2f',

            annot_kws={'size':10})

plt.show()
# correlation matrix: ignore score

us_df = cwu_df.loc[cwu_df['location']=='USA'].drop(['score','faculty_quality','education_quality'],axis=1) #

corr_df = us_df.dropna()

print("Number of entries after dropping NaN's: {} out of {}".format(len(corr_df),len(us_df)))

corrmat = corr_df.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat,

            cbar=True,

            annot=True,

            square = True,

            fmt='.2f',

            annot_kws={'size':10})

plt.show()