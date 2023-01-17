#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls

#Load datasets

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(2) #Check the dataset
train.shape 
train.describe(include="all")
#Correlation

train.corr()["SalePrice"]
corr=train[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();
#More Data visualizations
plt.figure(figsize=(12,6))
plt.scatter(x='GrLivArea', y='SalePrice', data=train)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
agg = train['MSZoning'].value_counts()[:10]
labels = list(reversed(list(agg.index )))
values = list(reversed(list(agg.values)))

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))
layout = dict(title='The general zoning classification', legend=dict(orientation="h"));


fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')



sns.violinplot(x='FullBath', y='SalePrice', data=train)
plt.title("Sale Price vs Full Bathrooms")
sns.violinplot( x="HalfBath",y="SalePrice", data=train)
plt.title("Sale Price vs Half Bathrooms");

#1st Floor in sq.feet
plt.scatter(train["1stFlrSF"],train.SalePrice, color='red')
plt.title("Sale Price vs. First Floor square feet")
plt.ylabel('Sale Price (in dollars)')
plt.xlabel("First Floor square feet");
plt.figure(figsize=(14,6))
plt.xticks(rotation=60) 
sns.barplot(x="Neighborhood", y = "SalePrice", data=train)
plt.title("Sale Price vs Neighborhood",fontsize=15 )
plt.figure(figsize=(14,6))
sns.barplot(x="TotRmsAbvGrd",y="SalePrice",data=train)
plt.title("Sale Price vs Number of rooms", fontsize=15);
plt.figure(figsize=(14,6))
sns.barplot(x="OverallQual",y="SalePrice",data=train)
plt.title("Sale Price vs 'Overall material and finish quality'", fontsize=15);

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train = train.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)
train.shape
#LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('None')
#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
    
#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    
#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : for all these categorical basement-related features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna('None')

#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

train["MasVnrType"] = train["MasVnrType"].fillna("None")
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)


#MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can remove it

train = train.drop(['Utilities'], axis=1)

#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
#Check remaining missing values if any 
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head()