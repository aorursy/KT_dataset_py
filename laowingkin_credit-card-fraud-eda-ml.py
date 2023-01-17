# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing
df = pd.read_csv('../input/creditcard.csv')

df.head()
df.describe()
df.isnull().any()
V = df[[col for col in df.columns if 'V' in col]]



f, ax = plt.subplots(ncols = 2, nrows = 14, figsize=(15,2*len(V.columns)))





for i, c in zip(ax.flatten(), V.columns):

    sns.distplot(V[c], ax = i)



f.tight_layout()



plt.figure(figsize=(15,10))



sns.heatmap(df.corr(), annot = True, fmt = '.1f')
min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df)



df_norm = pd.DataFrame(np_scaled, columns = df.columns)



df_norm.head()
df_melt = pd.melt(df_norm, id_vars=['Class'], value_vars=[col for col in df.columns if 'V' in col])



df_melt.iloc[0:1000000:100000,:]
plt.figure(figsize=(15,10))



f, ax = plt.subplots(4, figsize = (15,10*4))



g1 = ['V'+str(i) for i in range(1,8)]

g2 = ['V'+str(i) for i in range(8,15)]

g3 = ['V'+str(i) for i in range(15,22)]

g4 = ['V'+str(i) for i in range(22,29)]



df_melt_1 = df_melt[df_melt['variable'].isin(g1)]

df_melt_2 = df_melt[df_melt['variable'].isin(g2)]

df_melt_3 = df_melt[df_melt['variable'].isin(g3)]

df_melt_4 = df_melt[df_melt['variable'].isin(g4)]



sns.violinplot(x="variable", y="value", hue="Class", data=df_melt_1, ax = ax[0], split=True)

sns.violinplot(x="variable", y="value", hue="Class", data=df_melt_2, ax = ax[1], split=True)

sns.violinplot(x="variable", y="value", hue="Class", data=df_melt_3, ax = ax[2], split=True)

sns.violinplot(x="variable", y="value", hue="Class", data=df_melt_4, ax = ax[3], split=True)
