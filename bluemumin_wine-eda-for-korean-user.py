import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')
wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

wine['quality'] = wine['quality'].replace(3, 'low rank')
wine['quality'] = wine['quality'].replace(4, 'low rank')
wine['quality'] = wine['quality'].replace(5, 'low rank')
wine['quality'] = wine['quality'].replace(6, 'high rank')
wine['quality'] = wine['quality'].replace(7, 'high rank')
wine['quality'] = wine['quality'].replace(8, 'high rank')
wine.rename(columns = {'fixed acidity' : 'fixacid', 'volatile acidity' : 'volacid', 'citric acid' : 'citacid', 
         'residual sugar' : 'rsugar', 'chlorides' : 'salt', 
                    'free sulfur dioxide' : 'freedioxid', 'total sulfur dioxide' : 'totaldioxid'  },   inplace = True)
wine.head()
wine.describe()
sns.set(style = "white")

wine_x_columns = wine.columns[:-1]
cor = wine.corr()

mask = np.zeros_like(cor, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize = (15,9))
cmap = sns.diverging_palette(200, 10, as_cmap = True)
sns.heatmap(cor, mask = mask,cmap = cmap,center = 0,square = True, 
            linewidths = 0.5, cbar_kws = {"shrink" : 1},annot = True)
plt.title('wine data correlation', size = 20)
ax.set_xticklabels(wine_x_columns, size = 12)
ax.set_yticklabels(wine_x_columns, size = 12);
wine['fixacid'].hist(figsize = (9,6))
plt.title('fixacid',size = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('value',size = 18)
plt.ylabel('count',size = 18);
wine['volacid'].hist(figsize=(9,6))
plt.title('volacid',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['citacid'].hist(figsize=(9,6))
plt.title('citacid',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['rsugar'].hist(bins=20,figsize=(9,6))
plt.title('rsugar',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['salt'].hist(figsize=(9,6))
plt.title('salt',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['freedioxid'].hist(bins=20,figsize=(9,6))
plt.title('freedioxid',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['totaldioxid'].hist(bins=30,figsize=(9,6))
plt.title('totaldioxid',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['density'].hist(figsize=(9,6))
plt.title('density',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['pH'].hist(figsize=(9,6))
plt.title('ph',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['sulphates'].hist(figsize=(9,6))
plt.title('sulphates',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
wine['alcohol'].hist(figsize=(9,6))
plt.title('alcohol',size=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('value',size=18)
plt.ylabel('count',size=18);
quantile_ninety_five=2*wine.quantile(0.95) 

quantile_ninety_five # 0 ~ 10
wine = wine[wine['fixacid'] <= quantile_ninety_five[0]]
wine = wine[wine['volacid'] <= quantile_ninety_five[1]]
wine = wine[wine['citacid'] <= quantile_ninety_five[2]]
wine = wine[wine['rsugar'] <= quantile_ninety_five[3]]
wine = wine[wine['salt'] <= quantile_ninety_five[4]]
wine = wine[wine['freedioxid'] <= quantile_ninety_five[5]]
wine = wine[wine['totaldioxid'] <= quantile_ninety_five[6]]
wine = wine[wine['density'] <= quantile_ninety_five[7]]
wine = wine[wine['pH'] <= quantile_ninety_five[8]]
wine = wine[wine['sulphates'] <= quantile_ninety_five[9]]
wine = wine[wine['alcohol'] <= quantile_ninety_five[10]]

wine.describe()
wine.to_csv('wine2.csv', index=False)