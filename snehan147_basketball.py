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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from numpy.polynomial.polynomial import polyfit
raw_data = pd.read_csv("../input/college-basketball-dataset/cbb.csv")

raw_data.columns
data_subset = raw_data.copy()

data_subset.drop(['ADJOE','ADJDE','BARTHAG','ADJ_T','WAB','POSTSEASON','SEED'],axis=1,inplace=True)

data_subset['Win%'] = data_subset['W']/data_subset['G']



data_subset.drop(['W','G'],axis=1,inplace=True)
data_subset.corr()['Win%'].sort_values()[:-1] #This removes Win%, which would otherwise be 100% correlated with itself

data_subset.plot.scatter(x = 'EFG_O',y='Win%');
data_subset.plot.scatter(x = 'FTR',y='Win%');
data_subset.groupby(['CONF'])['Win%'].mean().sort_values(ascending=False).plot(kind='bar', figsize = (10,7));
data_subset[data_subset['Win%'] == data_subset['Win%'].min()]



data_subset[data_subset['Win%'] == data_subset['Win%'].max()]

data_subset['YEAR'] = data_subset['YEAR'].astype(str)



dummy_df = pd.get_dummies(data_subset)



standard_df = pd.DataFrame(StandardScaler().fit_transform(dummy_df), columns = dummy_df.columns)



standard_df = standard_df.drop('Win%',axis=1)



pca = PCA(n_components=3)



pca_df = pd.DataFrame(pca.fit_transform(standard_df))



pca_df.columns = ['Feature1','Feature2','Feature3']



pca_df.head()

abs(pd.Series(pca.components_[0],index = standard_df.columns)).sort_values(ascending=False)[:5]

abs(pd.Series(pca.components_[1],index = standard_df.columns)).sort_values(ascending=False)[:5]
abs(pd.Series(pca.components_[2],index = standard_df.columns)).sort_values(ascending=False)[:5]
regmodel = LinearRegression()



regmodel.fit(X = pca_df, y = dummy_df['Win%'])



outputs = regmodel.predict(pca_df)
# Fit with polyfit

b, m = polyfit(dummy_df['Win%'], outputs, 1)



plt.plot(dummy_df['Win%'], outputs, '.', alpha = 0.4)

plt.plot(dummy_df['Win%'], b + m * dummy_df['Win%'], '-')

plt.show()
results_df = data_subset[['TEAM','YEAR','Win%']].copy()



results_df['prediction'] = outputs



results_df['difference'] = results_df['Win%'] - results_df['prediction']

results_df.sort_values('difference')[:5]
results_df.sort_values('difference',ascending=False)[:5]
data_subset[data_subset.index == 367]
EFG_O_view = data_subset.groupby(round(data_subset['EFG_O'],0))['Win%'].mean()

EFG_D_view = data_subset.groupby(round(data_subset['EFG_D'],0))['Win%'].mean()

print(EFG_O_view[EFG_O_view.index == 49])

print(EFG_D_view[EFG_D_view.index == 55])

data_subset[data_subset.index == 292]



print(EFG_O_view[EFG_O_view.index == 48])

print(EFG_D_view[EFG_D_view.index == 57])
