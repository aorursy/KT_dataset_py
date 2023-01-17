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
import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('bmh')

df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

df.head()
df.columns
df.info()

print(df['price'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['price'], color='g', bins=100, hist_kws={'alpha': 0.4});

plt.savefig('price-dist.png')

# df['col_name'] = pd.to_numeric(df['col_name'], errors='coerce')



df_num = df.select_dtypes(include = ['float64', 'int64'])

plt.figure(figsize=(9, 8))

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

plt.savefig('int-column-dist.png')
df_num_corr = df_num.corr()['price'][:-1]

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

print("There are {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
corr = df_num.drop('price', axis=1).corr()

plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);

plt.savefig('heatmap.png')