# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



# Get the data into a dataframe

parts = pd.read_csv('../input/parts.csv')

part_categories = pd.read_csv('../input/part_categories.csv')
part_categories.info()
parts.info()
parts.columns = ['part_num', 'name', 'cat_id']

part_categories.columns = ['cat_id', 'cat_name']
left = parts

right = part_categories

parts_and_categories = pd.merge(left, right, on=["cat_id"], how='outer')

parts_and_categories['cat_name'].unique()

len(parts_and_categories['cat_name'].unique())
plt.suptitle('Part Category Popularity')

sns.countplot(x='cat_name', data=parts_and_categories)