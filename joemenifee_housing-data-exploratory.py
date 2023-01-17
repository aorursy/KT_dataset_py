# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



training = pd.read_csv('../input/train.csv')

testing = pd.read_csv('../input/test.csv')





# dataframe.size 

size = training.size 

print("*** size = ",size)  

# dataframe.shape 

shape = training.shape 

print("**** shape =",shape)  

# dataframe.ndim 

dimension = training.ndim 



print("**** Dimension",dimension)

print("---  Description -----")

training.describe(include='all')

print("*** Column names :", list(training.columns)) 

print("----- 5 samples -----")

training.sample(5)

list(training.columns) 

training['YrSold'].value_counts().plot.bar(title="Year Sold")
training.dtypes




print(training.isnull().sum())

categorical_features = {'Id', 'MSSubClass', 'MSZoning','Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'HeatingQC', 'CentralAir', 

                         'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars','GarageQual', 'GarageCond', 'PavedDrive', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'}

for feature in categorical_features:

    training[feature] = training[feature].astype("category")

    

#training.dtypes

training.get_dtype_counts()
training['TotRmsAbvGrd'].value_counts().plot.bar(title="Total Room Count")
training['BedroomAbvGr'].value_counts().plot.bar(title="Total Bedroom Count")
plt.figure(figsize=(25, 10))

training['YearBuilt'].value_counts().plot.bar(title="Year built")
import pylab 

print("*** SalePrice :")

plt.hist(training.SalePrice,bins='auto')

print("*** Log(SalePrice)")

plt.hist(np.log(training.SalePrice),bins=200)

print("Pylab")

pylab.hist(training.SalePrice, bins='auto')
plt.scatter(training.YrSold,training.SalePrice)
plt.boxplot(training.TotRmsAbvGrd)
import  seaborn as sb

plt.figure(figsize=(20, 8))

sb.distplot(training['SalePrice'], color='g', bins=30, hist_kws={'alpha': 0.4});
plt.figure(figsize=(20, 8))

sb.distplot(np.log(training['SalePrice']), color='g', bins=60, hist_kws={'alpha': 0.4});
print("Sale Price Distribution",training['SalePrice'].describe())
plt.figure(figsize=(20, 8))

sb.distplot(training['GrLivArea'], color='b', bins=60, hist_kws={'alpha': 0.4});
training_num = training.select_dtypes(include = ['float64', 'int64'])

training_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
from scipy.stats import skew

from scipy.stats import kurtosis

import matplotlib.pyplot as plt



#plt.style.use('ggplot')



#np.var(training_num)



#plt.hist(training_num, bins=60)



training_num.hist(bins=50,figsize=(20,15))

plt.show()
print("*** mean :  \n", np.mean(training_num))

print("*** var  :  \n", np.var(training_num))

print("*** skew :  \n",skew(training_num))

print("*** kurt :  \n",kurtosis(training_num))
training_num.plot.hist(alpha=0.5, bins=15, grid=True, legend=None)  # Pandas helper function to plot a hist. Uses matplotlib under the hood.

plt.xlabel("Feature value")

plt.title("Histogram")

plt.show()
# Second attemp to balance distribution using log

training_sqrt = training_num.apply(np.sqrt)   # pd.DataFrame.apply accepts a function to apply to each column of the data

training_sqrt.plot.hist(alpha=0.5, bins=15, grid=True, legend=None)

plt.xlabel("Feature value")

plt.title("Histogram of Square Root")

plt.show()
from pandas import scatter_matrix

sns.set()

feature = ['SalePrice','GrLivArea']

scatter_matrix(training_num[feature],figsize=(15,8))