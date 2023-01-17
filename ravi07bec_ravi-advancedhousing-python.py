#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_train.head(10)
df_train.shape
#New Column add
df_train['Age of Property'] = 2018-df_train['YearBuilt']
df_train['Age of Property'].describe()
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#Missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
#For cbind
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
from matplotlib import pyplot
a4_dims = (20, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.distplot(df_train['SalePrice']);
df_train[(df_train.SalePrice < 100000) & (df_train.SalePrice > 90000)].head(10)
#For ScatterPlot of Area vs Price
from matplotlib import pyplot
a4_dims = (20, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.regplot(ax=ax,x="LotArea", y="SalePrice", data=df_train[(df_train.LotArea < 50000)])
#For BoxPlot of Area vs Year
from matplotlib import pyplot
a4_dims = (20, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
fig=sns.boxplot(ax=ax,x="YearBuilt", y="SalePrice", data=df_train[(df_train.YearBuilt > 1990)])
fig.axis(ymin=0, ymax=500000);
plt.xticks(rotation=90);
#Multiple Subplots
from matplotlib import pyplot
a4_dims = (20, 8.27)
fig, ax1 = pyplot.subplots(figsize=a4_dims)
fig, ax2 = pyplot.subplots(figsize=a4_dims)
sns.boxplot(ax=ax1,x="YearBuilt", y="SalePrice", data=df_train[(df_train.YearBuilt > 1990)])
sns.distplot(df_train[(df_train.YearBuilt > 1990)].YearBuilt,ax=ax2) #Subset and then take only 1 column


#Find Unique Entries
b=df_train['YearBuilt'].unique()
print("Number of Unique entries is",b.shape)

#Factor Plot trend
sns.factorplot(data = df_train, x = 'YearBuilt', y = "SalePrice", 
               col = 'OverallQual',
               hue = 'GarageCars',
               palette = 'RdPu') 
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
g=sns.pairplot(df_train[cols], size = 2.5,diag_kind="kde")
g.set(yticklabels=[])
plt.show();