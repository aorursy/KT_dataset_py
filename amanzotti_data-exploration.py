%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import pandas as pd

import seaborn as sns

import numpy as np

import pylab

pylab.rcParams['figure.figsize'] = (12, 8)

plt.style.use('ggplot')

plt.rcParams['image.cmap'] = 'gray'

matplotlib.rcParams.update({'font.size': 14})
data = pd.read_csv('../input/train.csv')
data.head()
sns.distplot(data.SalePrice)
sns.distplot(np.log(data.SalePrice))
data.SalePrice = np.log(data.SalePrice)
col = pd.get_dummies(data).columns[np.abs(pd.get_dummies(data).corr()['SalePrice'])>0.4]
sns.heatmap(pd.get_dummies(data)[col].corr())
data.MSSubClass.unique()
data.MSSubClass.value_counts()
sns.countplot(data.MSSubClass)
sns.boxplot(x=data.MSSubClass, y=(data.SalePrice))
sns.boxplot(x=data.MSSubClass, y=np.log(data.SalePrice))
data.MSZoning.value_counts()
sns.countplot(data.MSZoning)
sns.boxplot(x=data.MSZoning, y=data.SalePrice)
sns.distplot(data.LotFrontage.dropna())
print (data.LotFrontage.mean())

print (data.LotFrontage.median())
np.mean(data.LotFrontage.isnull())
sns.regplot(x=data.LotFrontage.fillna(data.LotFrontage.median()), y=data.SalePrice);
sns.distplot(data.LotArea.dropna())
np.mean(data.LotArea.isnull())
sns.regplot(x=data.LotArea, y=data.SalePrice);
np.mean(data.Street.isnull())
data.Street.value_counts()
sns.boxplot(x=data.Street, y=data.SalePrice)
data.Alley.value_counts()
np.mean(data.Alley.isnull())
sns.boxplot(x=data.Alley, y=data.SalePrice)
data.LotShape.value_counts()
np.mean(data.LotShape.isnull())
sns.boxplot(x=data.LotShape, y=data.SalePrice)
print(data.LandContour.value_counts())

print(np.mean(data.LandContour.isnull()))

sns.boxplot(x=data.LandContour, y=data.SalePrice)
print(data.Utilities.value_counts())

print(np.mean(data.Utilities.isnull()))

sns.boxplot(x=data.Utilities, y=data.SalePrice)
print(data.LotConfig.value_counts())

print(np.mean(data.LotConfig.isnull()))

sns.boxplot(x=data.LotConfig, y=data.SalePrice)
print(data.LandSlope.value_counts())

print(np.mean(data.LandSlope.isnull()))

sns.boxplot(x=data.LandSlope, y=data.SalePrice)
print(data.Neighborhood.value_counts())

print(np.mean(data.Neighborhood.isnull()))

sns.boxplot(x=data.Neighborhood, y=data.SalePrice)
print(data.Condition1.value_counts())

print(np.mean(data.Condition1.isnull()))

sns.boxplot(x=data.Condition1, y=data.SalePrice)
print(data.Condition2.value_counts())

print(np.mean(data.Condition2.isnull()))

sns.boxplot(x=data.Condition2, y=data.SalePrice)
print(data.BldgType.value_counts())

print(np.mean(data.BldgType.isnull()))

sns.boxplot(x=data.BldgType, y=data.SalePrice)
print(data.HouseStyle.value_counts())

print(np.mean(data.HouseStyle.isnull()))

sns.boxplot(x=data.HouseStyle, y=data.SalePrice)
sns.distplot(data.OverallQual.dropna())

plt.show()

print(np.mean(data.OverallQual.isnull()))

sns.regplot(x=data.OverallQual, y=data.SalePrice);
sns.distplot(data.OverallCond.dropna())

plt.show()

print(np.mean(data.OverallCond.isnull()))

sns.regplot(x=data.OverallCond, y=data.SalePrice);
sns.boxplot(x=data.OverallCond, y=data.SalePrice)
sns.distplot(data.YearBuilt.dropna())

plt.show()

print(np.mean(data.YearBuilt.isnull()))

sns.regplot(x=data.YearBuilt, y=data.SalePrice);
sns.distplot(data.YearRemodAdd.dropna())

plt.show()

print(np.mean(data.YearRemodAdd.isnull()))

sns.regplot(x=data.YearRemodAdd, y=data.SalePrice);
print(data.RoofStyle.value_counts())

print(np.mean(data.RoofStyle.isnull()))

sns.boxplot(x=data.RoofStyle, y=data.SalePrice)
print(data.RoofMatl.value_counts())

print(np.mean(data.RoofMatl.isnull()))

sns.boxplot(x=data.RoofMatl, y=data.SalePrice)
print(data.Exterior1st.value_counts())

print(np.mean(data.Exterior1st.isnull()))

sns.boxplot(x=data.Exterior1st, y=data.SalePrice)
sns.distplot(data.GarageCars.dropna())

plt.show()

print(np.mean(data.GarageCars.isnull()))

sns.regplot(x=data.GarageCars, y=data.SalePrice);
sns.distplot(data['1stFlrSF'])

plt.show()

print(np.mean(data['1stFlrSF'].isnull()))

sns.regplot(x=data['1stFlrSF'], y=data.SalePrice);
sns.distplot(data['2ndFlrSF'])

plt.show()

print(np.mean(data['2ndFlrSF'].isnull()))

sns.regplot(x=data['2ndFlrSF'], y=data.SalePrice);
print(data.Exterior2nd.value_counts())

print(np.mean(data.Exterior2nd.isnull()))

sns.boxplot(x=data.Exterior2nd, y=data.SalePrice)