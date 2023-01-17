# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
# Any results you write to the current directory are saved as output.
pd.__version__
print(os.listdir('../input')[:2])
imcpusa0 = pd.read_excel('../input/Imports Merchandise Customs Price US seas. adj..xlsx', sheet_name=0)
imcpusa0.head()
imcpusa0.shape
imcpusa0 = imcpusa0.dropna(axis=0, how='all')
imcpusa0.head()
imcpusa0_corr = imcpusa0.corr()
imcpusa0_corr.head()
mask = np.zeros_like(imcpusa0_corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(imcpusa0_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
imcpusa0_corr[mask] = None
imcpusa0_corr.head()
imcpusa0_corr_melted = pd.melt(imcpusa0_corr.reset_index(), id_vars='index', value_vars=imcpusa0_corr.columns.tolist())
imcpusa0_corr_melted.head()
imcpusa0_corr_melted = imcpusa0_corr_melted.dropna()
imcpusa0_corr_melted.head()
imcpusa0_corr_melted = imcpusa0_corr_melted.sort_values('value')
imcpusa0_corr_melted.head()
imcpusa0_corr_melted.tail()
most_neg = ['Europe & Central Asia developing', 'High Income: Non-OECD']
sns.pairplot(imcpusa0.fillna(0).loc[:, most_neg]);
most_pos = ['Developing Countries', 'Middle Income Countries']
sns.pairplot(imcpusa0.fillna(0).loc[:, most_pos]);
fig, ax = plt.subplots(figsize=(15, 8))
imcpusa0.loc[:, most_neg].plot(ax=ax);
ax.set(xlabel='Year', ylabel='Price')
ax.grid()
fig, ax = plt.subplots(figsize=(15, 8))
imcpusa0.loc[:, most_pos].plot(ax=ax);
ax.set(xlabel='Year', ylabel='Price')
ax.grid()
imcpusa1 = pd.read_excel('../input/Imports Merchandise Customs Price US seas. adj..xlsx', sheet_name=1)
imcpusa1.head()
imcpusa1.shape
imcpusa1 = imcpusa1.dropna(axis=0, how='all')
imcpusa1.head()
imcpusa1.index.max()
months = pd.date_range(start='1991-01-01', end='2018-08-01', freq='MS')
months
imcpusa1.shape
imcpusa1.index = months
imcpusa1_corr = imcpusa1.corr()
mask = np.zeros_like(imcpusa1_corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(imcpusa1_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
imcpusa1_corr[mask] = None
imcpusa1_corr_melted = pd.melt(imcpusa1_corr.reset_index(), id_vars='index', value_vars=imcpusa1_corr.columns.tolist())
imcpusa1_corr_melted = imcpusa1_corr_melted.dropna()
imcpusa1_corr_melted = imcpusa1_corr_melted.sort_values('value')
imcpusa1_corr_melted.head()
most_neg = ['High Income: Non-OECD', 'Slovakia', 'Philippines', 'Turkey']
imcpusa1_corr_melted.tail()
most_pos = ['Europe & Central Asia developing', 'Turkey', 'Developing Countries', 'Middle Income Countries']
sns.pairplot(imcpusa1.fillna(0).loc[:, most_neg]);
imcpusa1.loc[:, most_neg].plot(figsize=(15, 10), grid=True, logy=True);
sns.pairplot(imcpusa1.fillna(0).loc[:, most_pos]);
imcpusa1.loc[:, most_pos].plot(figsize=(15, 10), grid=True, logy=True);
imcpusa2 = pd.read_excel('../input/Imports Merchandise Customs Price US seas. adj..xlsx', sheet_name=2)
imcpusa2.shape
imcpusa2.head()
imcpusa2 = imcpusa2.dropna(axis=0, how='all')
imcpusa2.head()

imcpusa2.index.max()
quarters = pd.date_range(start='1991-01-01', end='2018-09-30', freq='Q')
quarters
imcpusa2.shape
imcpusa2.index = quarters
imcpusa2_corr = imcpusa2.corr()
mask = np.zeros_like(imcpusa2_corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(imcpusa2_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
is_small_columns = imcpusa2_corr.where(imcpusa2_corr < -.6).notnull().any(axis=0)
is_small_rows = imcpusa2_corr.where(imcpusa2_corr < -0.6).notnull().any(axis=1)
neg_corrs2 = is_small_columns.loc[is_small_columns.values].index.tolist()
sns.pairplot(imcpusa2.fillna(0).loc[:, neg_corrs2]);
imcpusa2.loc[:, neg_corrs2].plot(figsize=(15, 10), grid=True);
