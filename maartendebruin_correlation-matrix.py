
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('../input/train.csv')

corrmat = train.corr()
plt.subplots(figsize=(20, 10))
sns.heatmap(corrmat, vmax=.9, square=True)
import seaborn as sns
import pandas as pd
import numpy as np
# Read data files
train = pd.read_csv('../input/train.csv')

# Create scatter plot of GrLivArea variable
var = 'PoolArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# Create scatter plot of GrLivArea variable
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice',xlim=(0,3500), ylim=(0, 700000))

# Create scatter plot of GrLivArea variable
var = 'LotArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice',xlim=(0,20000), ylim=(0, 700000))

# Create scatter plot of GrLivArea variable
var = 'LotFrontage'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var = 'Fireplaces'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var = 'GarageCars'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var = 'GarageArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


