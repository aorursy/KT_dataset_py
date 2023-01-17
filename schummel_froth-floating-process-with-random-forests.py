# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from IPython.display import display

from sklearn import metrics



# include fasti.ai libraries

from fastai.tabular import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from IPython.display import display

pd.set_option('display.max_columns', None) # display all columns

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv", parse_dates = True, index_col = 'date',decimal=',')
shape1 = df.shape

df = df.dropna()

shape2 = df.shape

if shape1 == shape2:

    print('Data contains no nan values.')

else:

    print('Data contains nan values.')
df.head()
plt.figure(figsize=(25,8))

plt.subplot(1, 2, 1)

plt.plot(df['% Iron Concentrate']);

plt.xlabel('Date')

plt.title('Iron Concentrate in %')

plt.subplot(1, 2, 2)

plt.plot(df['% Silica Concentrate']);

plt.xlabel('Date')

plt.title('Silica Concentrate in %')
sep_date = "2017-03-29 12:00:00"

ind_date = df.index<sep_date #boolean of earlier dates

df.drop(df.index[ind_date],inplace=True)

df.head(1)
plt.figure(figsize=(30, 25))

p = sns.heatmap(df.corr(), annot=True)
train, test = train_test_split(df, test_size=0.2)

x = train.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)

y = train['% Silica Concentrate']
model = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, max_features=None, n_jobs=-1)

model.fit(x,y)
y_hat = model.predict(x)

mse = metrics.mean_squared_error(y,y_hat)

print('Train Set')

print('RMSE:',math.sqrt(mse),'   R2:',model.score(x,y))
x_test = test.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)

y_test = test['% Silica Concentrate']

y_hat_test = model.predict(x_test)

mse_test = metrics.mean_squared_error(y_test,y_hat_test)

print('TEST Set')

print('RMSE:',math.sqrt(mse_test),'   R2:',model.score(x_test,y_test))
feat_importances = pd.Series(model.feature_importances_, index=df.columns[:-2])

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(df).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)

plt.show()
df_mean = df.copy()

mean_grpby = df_mean.groupby(['date']).mean() # calculate mean

std_grpby = df_mean.groupby(['date']).std() # calculate std

std_grpby = std_grpby.loc[:, (std_grpby != 0).any(axis=0)] # delete null columns (columns with zero variance)

std_grpby = std_grpby.add_prefix('STD_') # add prefix to column names

df_merge = pd.merge(mean_grpby, std_grpby, on='date') # merge both dataframes

df_merge.describe()
train, test = train_test_split(df_merge, test_size=0.2)

x_aver = train.drop(['% Silica Concentrate','% Iron Concentrate','STD_% Silica Concentrate','STD_% Iron Concentrate'], axis=1)

y_aver = train['% Silica Concentrate']
model = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, max_features=None, n_jobs=-1)

model.fit(x_aver,y_aver)
y_aver_hat = model.predict(x_aver)

mse = metrics.mean_squared_error(y_aver,y_aver_hat)

print('Train Set')

print('RMSE:',math.sqrt(mse),'   R2:',model.score(x_aver,y_aver))
x_aver_test = test.drop(['% Silica Concentrate','% Iron Concentrate','STD_% Silica Concentrate','STD_% Iron Concentrate'], axis=1)

y_test = test['% Silica Concentrate']

y_hat_test = model.predict(x_aver_test)

mse_test = metrics.mean_squared_error(y_test,y_hat_test)

print('TEST Set')

print('RMSE:',math.sqrt(mse_test),'   R2:',model.score(x_aver_test,y_test))