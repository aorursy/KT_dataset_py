#I will keep updating this kernel. 

#V1 contains visualizing the data set. types of data. Missing values, correlation heatmap 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import missingno as msno

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Let look at the columns in the training data. 

train.head(3).transpose()
#types of columns

train.dtypes
#lets make a bar chart for type of data

bdataTypeDf = pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})

fig,ax = plt.subplots()

fig.set_size_inches(10,7.5)

sns.barplot(data=bdataTypeDf,x="variableType",y="count",ax=ax,color="#0b53c6")

ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")
#let look at the target variable

train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
#missing values 

number_of_columns_contains_NA = train.isnull().any().sum()

print( number_of_columns_contains_NA , "Columns has missing Values in training dataset")

number_of_columns_contains_NA = test.isnull().any().sum()

print( number_of_columns_contains_NA , "Columns has missing Values in test dataset")
#lets look at the ratio of missing values in the training dataset

import missingno as msno



missingValueColumns = train.columns[train.isnull().any()].tolist()

msno.bar(train[missingValueColumns],\

            figsize=(30,8),color="#34495e",fontsize=15,labels=True,)
#heatmaps between missing values columns

msno.heatmap(train[missingValueColumns],figsize=(20,20))
#filling missing values will be covered in the next version. stay tuned
#lets look at some relationship between target variable and other variables

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#lets look at the relationship with categorical values

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#lets look at the year built and target variable

var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#Co-relation heatmap

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();