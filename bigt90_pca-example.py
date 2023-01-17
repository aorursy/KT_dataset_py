import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import datetime as dt

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_dji = pd.read_csv(os.path.join(dirname, 'djia.csv'))

df_comp = pd.read_csv(os.path.join(dirname, 'stock_prices.csv'))

df_comp['Date'] = pd.to_datetime(df_comp['Date'], dayfirst=True)

df_dji['Date'] = pd.to_datetime(df_dji['Date'], dayfirst=True)

df_comp = df_comp[df_comp.Date > dt.datetime(2019,3,19)].reset_index(drop=True)

df_dji = df_dji[df_dji.Date > dt.datetime(2019,3,19)].reset_index(drop=True)
sc = StandardScaler()

X = df_comp.iloc[:,1:].values

y = df_dji.iloc[:,1].values

X_std = sc.fit_transform(X)

y_std = sc.fit_transform(y.reshape(-1, 1))

y_std.shape
from sklearn.decomposition import PCA





pca = PCA(n_components=None)

pca.fit(X_std)

cum_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(12,8))

plt.bar(range(1, X_std.shape[1]+1), pca.explained_variance_ratio_, 

        alpha=0.8,

        align='center',

        label='individual explained variance')

plt.step(range(1, X_std.shape[1]+1), cum_var, 

         where='mid', 

         label='cumulative explained variance')

plt.legend(fontsize=12)

plt.xlabel('Principal components', fontsize=14)

plt.ylabel('Explained variance ratio', fontsize=14)

plt.grid(linestyle = '-.')

plt.show()
pca = PCA(n_components=5)

X_pca = pca.fit_transform(X_std)

pca.explained_variance_ratio_
print('First 5 components explain %.2f%% variance.' % (100*pca.explained_variance_ratio_.sum()))
pca.components_[2]
all_comp = df_comp.columns[1:].values

print('Largest contribution in each principal component')

print('-'*35)

for i in range(pca.components_.shape[0]):

    print('PC{0}: {1}'.format(i+1, all_comp[abs(pca.components_[i]).argmax()]))
print('PC and DJIA correlation')

print('-'*25)

corr_matr = np.corrcoef(y_std.T, X_pca.T)

for i in range(5):

    print('PC%i: %.3f' % (i+1, corr_matr[0,1:][i]))

np.round(corr_matr, 3)