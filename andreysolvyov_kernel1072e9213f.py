# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/12312312336756859555/SearchAds.csv")
temp_df = data.fillna(0)
data
objects = []
types = []
for j in temp_df.columns:
    types.append(temp_df[[j]].dtypes[0])
    if temp_df[[j]].dtypes[0] == 'object':
        objects.append(j)
print(set(types))
print(objects)
temp_df['Impressions'] = temp_df['Impressions'].map(lambda x: int(str(x).replace(',','')))
temp_df['Taps'] = temp_df['Taps'].map(lambda x: int(x))
temp_df['Installs'] = temp_df['Installs'].map(lambda x: int(x))
temp_df['Spend'] = temp_df['Spend'].map(lambda x: float(str(x).replace('$','')))
temp_df
df = temp_df
df_t = df[['Search Term', 'Taps', 'Spend']]
df_t
df_t[(df_t['Spend'] != 0) & (df_t['Taps'] == 0)]
df['nd_T'] = df['Taps'] > 5
df_t = df_t.iloc[df[df['nd_T'] == True].index]
df_t['Convers_Taps'] = df_t['Taps']/df_t['Spend']
df_t.sort_values(by='Convers_Taps')[::-1]
df_t.sort_values(by='Convers_Taps')[::-1].plot(kind='bar', x = 'Search Term', y = 'Convers_Taps', figsize = (25, 5));
for alpha in [0.5, 0.75, 0.9]:
    df_t['alpha_T ' + str(alpha)] = df_t['Convers_Taps'] >= max(df_t['Convers_Taps'])*alpha
df_t
print(df_t[df_t['alpha_T 0.9']]['Search Term'].values)
from scipy.stats import spearmanr
df['nd'] = (df['Impressions'] > df['Impressions'].median())*(df['Installs'] > df['Installs'].median())
df_corr = df.iloc[df[df['nd'] == True].index]
r = spearmanr(df_corr['Impressions'], df_corr['Taps'])
print('Pearson correlation for Impressions-Taps:', r[0], 'p-value:', r[1])
r = spearmanr(df_corr['Installs'], df_corr['Taps'])
print('Pearson correlation for Installs-Taps:', r[0], 'p-value:', r[1])
df_g = df[['Search Term', 'Impressions', 'Installs']]
df_g = df_g[df['Impressions'] > 50]
df_g['Convers_I'] = df_g['Installs']/df_g['Impressions']
df_g
df_g.sort_values(by='Convers_I')[::-1].plot(kind='bar', x = 'Search Term', y = 'Convers_I', figsize = (25, 5));
for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    print('Уровень значимости: ', alpha)
    print('Соответствующий уровень конверсии: ', max(df_g['Convers_I'])*alpha)
    print()
for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    df_g['alpha_C_I ' + str(alpha)] = df_g['Convers_I'] >= max(df_g['Convers_I'])*alpha
df_g
print(df_g[df_g['alpha_C_I 0.5']]['Search Term'].values)