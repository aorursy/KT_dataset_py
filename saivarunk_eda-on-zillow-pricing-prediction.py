# Importing required dependent libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
# Import dataset



properties = pd.read_csv("../input/properties_2016.csv")

train = pd.read_csv("../input/train_2016_v2.csv")

submission = pd.read_csv("../input/sample_submission.csv")
missing_df = properties.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
missing_df.column_name.values
missing_df.head()
missing_df.sort_values(by="missing_count")
missing_df.column_name.values
msno.bar(properties)