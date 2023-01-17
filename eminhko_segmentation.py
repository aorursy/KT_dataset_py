# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load packages
print(__doc__)
%matplotlib inline
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#Load data
df_train = pd.read_csv('/kaggle/input/transactions/Transactions.csv')

print(df_train.head())
#Examine data
print(df_train.describe())
#Replace missing values with zero
df_train.fillna(0,inplace = True)
print(df_train.head())
#Examine data after remove missing values
print(df_train.describe())
#Examine negative values
negative_cells = df_train<0
print(negative_cells.sum())
df_train = df_train[negative_cells.sum(axis=1) < 1]
np.shape(df_train)
#Total spending for each category
df_sum = df_train[['Category_' + str(i) for i in [1,2,3,4,5,6]]].sum(axis=1)
print(df_sum.head())
#Examine total spendings
print(df_sum.describe())
# Basic plot of total spendings
plt.figure(figsize=(5,10))
plt.boxplot(x=df_sum.values, sym='ko')
plt.show()
#Find percentile values for total spendings
print(df_sum.min())
print(np.percentile(df_sum, 90))
print(np.percentile(df_sum, 95))
print(np.percentile(df_sum, 99))
print(df_sum.max())
df_train['sum'] = df_sum

df_train = df_train[df_train['sum'] <= np.percentile(df_train['sum'], 99)]
df_train = df_train[df_train['sum'] >0]

#Examine data after eliminate outliers
print(df_train.describe())
#After outlier detection, let's see how graph will look like
plt.figure(figsize=(5,10))
plt.boxplot(df_train['sum'].values, sym='ko')
plt.show()
#Category columns
col_cat = [ u'Category_1', u'Category_2', u'Category_3', u'Category_4', u'Category_5', u'Category_6']

df_train_cat = df_train[col_cat]

#Calculate correlation matrix
corr_matrix = df_train_cat.corr()

#Use seaborn for heatmap plot
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,square=True)
plt.show()
df_train_cat.columns = ['Meat-Fish','Dairy Products','Frozen Foods','Vegetables','Fruits','Personal Care']

#Calculate correlation matrix
corr_matrix = df_train_cat.corr()

#Use seaborn for heatmap plot
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,square=True)
plt.show()
#Merge Category4 and Category5
df_train_cat['Vegs-Fruits'] = df_train_cat['Vegetables'] + df_train_cat['Fruits']

df_train_cat.drop(['Vegetables', 'Fruits'], axis=1, inplace=True)
df_train_cat = df_train_cat[['Meat-Fish', 'Dairy Products', 'Frozen Foods', 'Vegs-Fruits', 'Personal Care']]

#Calculate correlation matrix
corr_matrix = df_train_cat.corr()

#Use seaborn for heatmap plot
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,square=True)
plt.show()
#Get floating divison of dataframe
df_train_norm = df_train_cat.div(df_train_cat.sum(axis=1), axis=0)
df_train_norm.head()
#Import Gaussian Mixture package
from sklearn.mixture import GaussianMixture

# We had 5 components
gausmix = GaussianMixture(n_components=5, random_state=0)
gausmix.fit(df_train_norm)
means = gausmix.means_

print(means)
#Plot means for components
plt.figure(figsize=(12,8))
plt.plot(means.transpose())

plt.xticks(np.arange(5), df_train_norm.columns)
plt.show()
[df_train_norm.iloc[1,:]][0]
#Prediction of most likely path for customer with index 1
path = gausmix.predict([df_train_norm.iloc[1,:]])[0]

plt.figure(figsize=(12,8))           
plt.plot(means[path,:].transpose(), linewidth=6, label = 'Likely path')
plt.plot(df_train_norm.iloc[1,:].values, linewidth = 8, label='Customer with index 1')
plt.plot(means.transpose())
plt.xticks(np.arange(5), df_train_norm.columns)
plt.legend()
plt.show()
[df_train_norm.iloc[15,:]][0]
path = gausmix.predict([df_train_norm.iloc[15,:]])[0]

plt.figure(figsize=(12,8))           
plt.plot(means[path,:].transpose(), linewidth=6, label = 'Likely path')
plt.plot(df_train_norm.iloc[15,:].values, linewidth = 8, label='Customer with index 15')
plt.plot(means.transpose())
plt.xticks(np.arange(5), df_train_norm.columns)
plt.legend()
plt.show()