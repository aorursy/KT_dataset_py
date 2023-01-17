# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



pd.options.display.max_rows = 1000

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)
train.head()
test.head()
train.duplicated().value_counts()
train.dtypes
train['MSSubClass'] = train['MSSubClass'].astype('category',copy=False)

train['OverallQual'] = train['OverallQual'].astype('category',copy = False, ordered = True)

train['OverallCond'] = train['OverallCond'].astype('category',copy = False, ordered = True)

train['CentralAir'] = train['CentralAir'].astype('category',copy = False)

#Leave the years as it is for now



test['MSSubClass'] = test['MSSubClass'].astype('category',copy=False)

test['OverallQual'] = test['OverallQual'].astype('category',copy = False, ordered = True)

test['OverallCond'] = test['OverallCond'].astype('category',copy = False, ordered = True)

test['CentralAir'] = test['CentralAir'].astype('category',copy = False)

missingCount = train.isnull().sum()

missingCount = missingCount[missingCount > 0]

print(missingCount)

plt.figure(figsize=(12,8))

ax = missingCount.plot(kind='bar')

ax.set_xlabel("Features")

ax.set_title("Features vs Missing value count")

ax.set_ylabel("MIssing value count")



#Creaing labels

for i in ax.patches:

    ax.text(i.get_x()-0.1, i.get_height()+5, str(round((i.get_height()/train.shape[0])*100,2))+'%')
train.drop(['LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

test.drop(['LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
train.loc[:,'MasVnrType'] = train.loc[:,'MasVnrType'].fillna('None')

train.loc[:,'MasVnrArea'] = train.loc[:,'MasVnrArea'].fillna(0)

train.loc[:,'BsmtQual'] = train.loc[:,'BsmtQual'].fillna('No')

train.loc[:,'BsmtCond'] = train.loc[:,'BsmtCond'].fillna('No')

train.loc[:,'BsmtExposure'] = train.loc[:,'BsmtExposure'].fillna('No')

train.loc[:,'BsmtFinType1'] = train.loc[:,'BsmtFinType1'].fillna('No')

train.loc[:,'BsmtFinType2'] = train.loc[:,'BsmtFinType2'].fillna('No')

train.loc[:,'Electrical'] = train.loc[:,'Electrical'].fillna('SBrkr')

train.loc[:,'GarageType'] = train.loc[:,'GarageType'].fillna('No')

train.loc[:,'GarageFinish'] = train.loc[:,'GarageFinish'].fillna('No')

train.loc[:,'GarageQual'] = train.loc[:,'GarageQual'].fillna('No')

train.loc[:,'GarageCond'] = train.loc[:,'GarageCond'].fillna('No')





test.loc[:,'MasVnrType'] = test.loc[:,'MasVnrType'].fillna('None')

test.loc[:,'MasVnrArea'] = test.loc[:,'MasVnrArea'].fillna(0)

test.loc[:,'BsmtQual'] = test.loc[:,'BsmtQual'].fillna('No')

test.loc[:,'BsmtCond'] = test.loc[:,'BsmtCond'].fillna('No')

test.loc[:,'BsmtExposure'] = test.loc[:,'BsmtExposure'].fillna('No')

test.loc[:,'BsmtFinType1'] = test.loc[:,'BsmtFinType1'].fillna('No')

test.loc[:,'BsmtFinType2'] = test.loc[:,'BsmtFinType2'].fillna('No')

test.loc[:,'Electrical'] = test.loc[:,'Electrical'].fillna('SBrkr')

test.loc[:,'GarageType'] = test.loc[:,'GarageType'].fillna('No')

test.loc[:,'GarageFinish'] = test.loc[:,'GarageFinish'].fillna('No')

test.loc[:,'GarageQual'] = test.loc[:,'GarageQual'].fillna('No')

test.loc[:,'GarageCond'] = test.loc[:,'GarageCond'].fillna('No')
train.dtypes
qualitative = [f for f in train.columns if train.dtypes[f] == 'object' or train.dtypes[f] == 'bool' or train.dtypes[f].name == 'category']

quantitative = list(set(train.columns) - set(qualitative))

#Removing Id and Sale Price

quantitative.remove('Id')

quantitative.remove('SalePrice')

print(qualitative)

print(quantitative)
train.SalePrice.describe()
sns.distplot(train.SalePrice)
#Skew and kurtosis

print('Skewness: %f ' %train.SalePrice.skew())

print('Kurtosis: %f ' %train.SalePrice.kurt())
plt.figure(1)

plt.title('Log normal fit')

sns.distplot(train.SalePrice, kde=False, fit = st.lognorm)

plt.figure(2)

plt.title('JohnsonSu fit')

sns.distplot(train.SalePrice, kde=False, fit = st.johnsonsu)

plt.figure(3)

plt.title('Normal fit')

sns.distplot(train.SalePrice, kde=False, fit = st.norm)

#Checking for any 0 values before transformation

(train.SalePrice == 0).value_counts()

train['SalePriceLog'] = train.SalePrice.apply(lambda x: np.log(x))

sns.distplot(train.SalePriceLog)
data = pd.melt(train, value_vars=quantitative)

sns.set()

featureDistribution = sns.FacetGrid(data, col="variable",  col_wrap=2, sharex=False, sharey=False)

featureDistribution = featureDistribution.map(sns.distplot, "value")
def pairplot(x, y, **kwargs):

    ax = plt.gca()

    ts = pd.DataFrame({'time': x, 'val': y})

    ts = ts.groupby('time').mean()

    ts.plot(ax=ax)

plt.xticks(rotation=90)

    

data = pd.melt(train, id_vars=['SalePrice'], value_vars=quantitative)

featureDistribution = sns.FacetGrid(data, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)

featureDistribution = featureDistribution.map(pairplot, "value", "SalePrice")
data = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)

featureDistribution = sns.FacetGrid(data, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)

featureDistribution = featureDistribution.map(sns.boxplot, "value", "SalePrice")
#Gets Pearson correlation

correlation = train.corr()

plt.figure(figsize=(12,10))

sns.heatmap(correlation)
k = 11 #SalePriceLog would also be included and should be removed

cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].T)

sns.set(font_scale=2)

plt.figure(figsize=(15,15))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 20}, yticklabels=cols.values, xticklabels=cols.values)
quanFeatures = ['GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

qualFeatures = ['OverallQual','BsmtQual']

sns.set()

sns.pairplot(train[quanFeatures])
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
train.sort_values(by = 'TotalBsmtSF', ascending = False)[:1]
train = train.drop(train[train['Id'] == 333].index)
#Checking for any 0 values before transformation

print((train.GrLivArea == 0).value_counts())

print((train.TotalBsmtSF == 0).value_counts())



print((test.GrLivArea == 0).value_counts())

print((test.TotalBsmtSF == 0).value_counts())
train['HasBsmt'] = train.TotalBsmtSF.apply(lambda x: 1 if(x > 0) else 0)

test['HasBsmt'] = test.TotalBsmtSF.apply(lambda x: 1 if(x > 0) else 0)
train['GrLivAreaLog'] = train.GrLivArea.apply(lambda x: np.log(x))

train['TotalBsmtSFLog'] = train.TotalBsmtSF.apply(lambda x: np.log(x))



print(np.isinf(train.TotalBsmtSFLog).sum())

train.TotalBsmtSFLog = train.TotalBsmtSFLog.replace([np.inf, -np.inf], 0)

print(np.isinf(train.TotalBsmtSFLog).sum())



test['GrLivAreaLog'] = test.GrLivArea.apply(lambda x: np.log(x))

test['TotalBsmtSFLog'] = test.TotalBsmtSF.apply(lambda x: np.log(x))



print(np.isinf(test.TotalBsmtSFLog).sum())

test.TotalBsmtSFLog = test.TotalBsmtSFLog.replace([np.inf, -np.inf], 0)

print(np.isinf(test.TotalBsmtSFLog).sum())

#Checking the normality befor and after transformation

plt.figure(1)

plt.figtext(0.4,1,'GrLivArea')

st.probplot(train['GrLivArea'], plot=plt)

plt.figure(2)

plt.figtext(0.4,1,'GrLivAreaLog')

st.probplot(train['GrLivAreaLog'], plot=plt)
#Checking the normality befor and after transformation

plt.figure(1)

plt.figtext(0.4,1,'TotalBsmtSF')

st.probplot(train['TotalBsmtSF'], plot=plt)

plt.figure(2)

plt.figtext(0.4,1,'TotalBsmtSFLog')

st.probplot(train[train.TotalBsmtSFLog > 0]['TotalBsmtSFLog'], plot=plt)
#Checking the normality befor and after transformation

plt.figure(1)

plt.figtext(0.4,1,'SalePrice')

st.probplot(train['SalePrice'], plot=plt)

plt.figure(2)

plt.figtext(0.4,1,'SalePriceLof')

st.probplot(train['SalePriceLog'], plot=plt)
quanFeatures.remove('TotalBsmtSF')

quanFeatures.remove('GrLivArea')

quanFeatures.extend(['TotalBsmtSFLog','GrLivAreaLog'])

qualFeatures.append('HasBsmt')

final_data = train.loc[:,qualFeatures + quanFeatures ]

print(final_data.head())

y = train.loc[:,'SalePriceLog']

print(y)



final_test_set = test.loc[:, qualFeatures + quanFeatures]
final_data = pd.get_dummies(final_data)

final_test_set = pd.get_dummies(final_test_set)

final_data

trainX, testX, trainY, testY = train_test_split(final_data,y)
linearModel = LinearRegression().fit(trainX,trainY)

#Coefficient of determination of R^2 - should be as closet to 1 as possible

print("Coefficient of determination: %f" %linearModel.score(trainX,trainY))

print("Intercept: %f" %linearModel.intercept_)

print("Coefficients: ", linearModel.coef_)

print(list(zip(qualFeatures + quanFeatures, linearModel.coef_)))
#Checking correlation on the training data

predY = linearModel.predict(trainX)

#Checking correlation to get estimate of accuracy

np.corrcoef(predY, trainY)
print(metrics.mean_absolute_error(trainY, predY)) #mean absolute error (mae)

print(metrics.mean_squared_error(trainY, predY)) #mean square error (mse)

print(np.sqrt(metrics.mean_squared_error(trainY, predY))) #root mean square error (rmse)

print(np.sqrt(metrics.mean_squared_log_error(trainY, predY))) #mean square logarithmic error (msle)
#Checking correlation on the testing data

predY = linearModel.predict(testX)

#Checking correlation to get estimate of accuracy

np.corrcoef(predY, testY)
print(metrics.mean_absolute_error(testY, predY)) #mean absolute error (mae)

print(metrics.mean_squared_error(testY, predY)) #mean square error (mse)

print(np.sqrt(metrics.mean_squared_error(testY, predY))) #root mean square error (rmse)

print(np.sqrt(metrics.mean_squared_log_error(testY, predY))) #mean square logarithmic error (msle)
plt.scatter(testY,predY)
print(np.where(np.isnan(final_test_set)))
final_test_set.loc[1116,'GarageCars'] = 0

final_test_set.loc[660,'TotalBsmtSFLog'] = 0
final_test_y = linearModel.predict(final_test_set)

sub_df = pd.DataFrame({'Id':test.Id,'SalePrice':np.exp(final_test_y)})

sub_df.head(100)
sub_df.to_csv("housing.csv", index=False)
from IPython.display import HTML



def create_download_link(title = "Download CSV file", filename = "housing.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='housing.csv')