# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt, rcParams, rc
import seaborn as sns

#For code reproducibility and stability
np.random.seed(42)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('ggplot')
# Any results you write to the current directory are saved as output.
#load and import data
df=pd.read_csv('../input/housetrain.csv')
#show the information of all the columns
df.info(verbose=True)
#print data shape
print("The Dimension of this data is: {}".format(df.shape))
#Now let's look at the data and print the first 5 columns
df.head()
#plot a scatter plot for the data
plt.figure(figsize=(20,10))
plt.scatter(df.YearBuilt, df.SalePrice)
plt.xlabel("Year Built", fontsize=13)
plt.ylabel("Sale Price", fontsize=13)
plt.show()
plt.figure(figsize=(20,10))
plt.scatter(df.GrLivArea, df.SalePrice)
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('Sale price', fontsize=13)
plt.show()
#Remove outliers
df=df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

#Plot it again for validation
plt.figure(figsize=(20,10))
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#seaborn to plot histogram
sns.distplot(df['SalePrice'])
print('Skewness is: %f' % df['SalePrice'].skew())
print('Kurtosis is: %f' % df['SalePrice'].kurt())
corrolation_map=df.corr()
f, ax=plt.subplots(figsize=(20,10))
sns.heatmap(corrolation_map, vmax=1, square=True)

