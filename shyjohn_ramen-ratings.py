# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the data file
df = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
df.head()
df.info()
df['Stars'].unique()
unrated_idx = df.loc[df['Stars'] == 'Unrated'].index.to_list()
for i in range(len(unrated_idx)): 
    df.at[unrated_idx[i], 'Stars'] = 0
df['Stars'].unique()
df['Stars'] = pd.to_numeric(df['Stars'])
df['Top Ten'].unique()
df['Style'].unique()
df.loc[df['Style'].isna()]
# By researching, the products are sold in packs. 
df.at[2152, 'Style'] = 'Pack'
df.at[2442, 'Style'] = 'Pack'
x = df['Country'].sort_values().unique()
y = df.groupby('Country')['Stars'].count().to_list()

plt.rcParams["figure.figsize"] = [18,12]
plt.title('Number of Products by Country')
plt.pie(y, labels=x)
plt.xticks(rotation='vertical')
plt.show()
plt.rcParams["figure.figsize"] = [16,9]
df.boxplot(column='Stars',by='Country')

plt.title('Distribution of Ratings per Country')
plt.xticks(rotation='vertical')
plt.show()
zero_ratings = df[df['Stars'] == 0].groupby('Country')['Stars'].count().to_dict()

plt.rcParams["figure.figsize"] = [16,9]
plt.title('Number of Zero Ratings by Country')
plt.bar(zero_ratings.keys(), zero_ratings.values())
plt.xticks(rotation='vertical')
plt.show()
