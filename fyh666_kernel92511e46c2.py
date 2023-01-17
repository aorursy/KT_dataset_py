import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

# the module includes Statistical functions

from scipy import stats

from scipy.stats import norm

# the module to preprocess the dataset

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.metrics import mean_absolute_error

# just ignore the warning information and don't show them

import warnings

warnings.filterwarnings("ignore")

# Use it to replace AI"plt"
train = pd.read_csv('../input/train.csv')

print(train)

pd.set_option('max_colwidth',100)
print(train.columns)
sheet_1 = pd.DataFrame({'Variable': ['GrLivArea', 'LotArea', 'Neighborhood', 'OverallQual', 'YearBuilt', '1stFlrSF', 'TotalBsmtSF', 'TotRmsAbvGrd', 'FullBath', 'CentralAir', 'GarageCars', 'GarageCars'],

        'Segment': [1, 1, 2, 0, 0, 1, 1, 0, 0, 0, 1, 1],

        'Data Type': [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],

        'Comments': ['Above grade (ground) living area square', 'Lot size in square feetStreet', 'Physical locations within Ames city limits', 'Rates the overall material and finish of the house', 'Original construction dateYearRemodAdd', 'First Floor square feet', 'Total square feet of basement area Heating', 'Total rooms above grade (does not include bathrooms)', 'Basement full bathrooms', 'Central air conditioning', 'Size of garage in square feet GarageQual', 'Size of garage in car capacityCar']})

sheet_1
print(train['SalePrice'].describe())
p = sns.distplot(train['SalePrice'],color = 'purple', fit = norm, axlabel = 'SalePrice',label = 'Saleprice', hist = True)

fig = plt.figure()

p.set_title('SalePrice distribution')

p.set(xlabel='SalePrices')

p.set(ylabel='Frequency')

# return an object

print(p)

res = stats.probplot(train['SalePrice'], plot = plt)
sheet_2 = pd.DataFrame({'concept\\range': ['Kurtosis', 'Skewness'],

                        '< 0': ['Flatter than the peak of Normal Distribution', 'Negative deviation is larger and the tail is in the left'],

                        '= 0': ['Of the same steepness witn the oeak of Normal Distribution', 'Distribution form is the same as Normal Distritution'],

                        '> 0': ['More steep than the peak of Normal Distribution', 'Positive deviation is larger and the tail is in the right']})

sheet_2
sheet_2.to_excel('sheet_2.xlsx', sheet_name = 'Sheet_2')
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
data = pd.concat([train['SalePrice'], train['GrLivArea']],axis = 1)

data.plot.scatter(x = 'GrLivArea', y = 'SalePrice', xlim = (0,6000), ylim = (0,800000),title = 'the relationship between SalePrice and GrLivArea')
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis = 1)

data.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice', xlim = (0,6000), ylim = (0,800000),title = 'the relationship between SalePrice and TotalBsmtSF')
data = pd.concat([train['SalePrice'],train['LotArea']],axis = 1)

data.plot.scatter(x = 'LotArea', y = 'SalePrice', xlim = (0, 25000), ylim = (0,800000),title = 'the relationship between SalePrice and LotArea')
data = pd.concat([train['SalePrice'], train['1stFlrSF']], axis = 1)

data.plot.scatter(x = '1stFlrSF', y = 'SalePrice', xlim = (0,6000), ylim = (0,800000),title = 'the relationship between SalePrice and 1stFlrSF')
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis = 1)

f, ax = plt.subplots(figsize = (35, 25))

fig = sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = data,)

fig.set(ylim=(0,1000000))
data = pd.concat([train['SalePrice'], train['OverallQual']], axis = 1)

f, ax = plt.subplots(figsize = (15, 10))

fig = sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = data, hue ='OverallQual')

fig.set(xlim=(0,10))
data = pd.concat([train['SalePrice'],train['Neighborhood']], axis = 1)

f, ax = plt.subplots(figsize = (15, 10))

fig = sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = data, hue = 'Neighborhood')

fig.set(ylim = (0,800000))
# get the correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize = (10, 8))

sns.heatmap(corrmat, vmax = 0.8, linecolor = 'black', square = 'True' )
names = ['CentralAir', 'Neighborhood']

# classify the type

for x in names:

    label = preprocessing.LabelEncoder()

    train[x] = label.fit_transform(train[x])

corrmat = train.corr()

f, ax = plt.subplots(figsize = (10,8))

sns.heatmap(corrmat, vmax = 0.8, linecolor = 'black',square = True)
k = 10

f, ax = plt.subplots(figsize = (10,8))

# get ten most relavant ones

cols  = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

# get the correlation matrix

cm = np.corrcoef(train[cols].values.T)

# set the font

sns.set(font_scale = 1.25)

hm = sns. heatmap(cm, cbar = True , annot = True, square = True, fmt = '.2f', annot_kws = {'size':10}, yticklabels = cols.values,xticklabels = cols.values)

plt.show()
sheet_3 = pd.DataFrame({'Variable': ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'],

        'Segment': [1, 1, 1, 1, 1, 0, 0],

        'Data Type': [0, 1, 0, 0, 0, 0, 1],

        'Comments': ['Above grade (ground) living area square', 'Rates the overall material and finish of the house', 'Size of garage in car capacityCar', 'Total square feet of basement area Heating', 'Basement full bathrooms', 'Total rooms above grade (does not include bathrooms)', 'Original construction dateYearRemodAdd']})

sheet_3
sheet_3.to_excel('sheet_3.xlsx', sheet_name = 'Sheet_3')
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.pairplot(train[cols], size = 5)

plt.show()
sheet_4 = pd.DataFrame({'Mechanism': ['Missing Completely at Random(MCAR)', 'Missing at Random(MAR)', 'Not Missing at Random(NMAR)'],

                        'Comment': ['The missing data has nothing to do with complete variables and incomplete variables', 'The data deficency just depends on the complete variables', 'The data deficency just depends on the incomplete variables and can not be ignored']})

sheet_4
sheet_4.to_excel('sheet_4.xlsx', sheet_name = 'Sheet_4')
# count the number of the missing data of every variable and sort them in descending arrangement.

total = train.isnull().sum().sort_values(ascending = False)

# count the percentage of missing data

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

# show in a DataFrame,keys is just used to control the index

missing_data = pd.concat([total, percent], axis = 1,keys = ['Total', 'Percent'])

missing_data.head(20)
train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

# check if have delt with all missing data

train.isnull().sum().max()
# to get the mean and variance of normal distribution

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])

low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][: 10]

high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]

print('low of the distribution:', low_range)

print('high of the distribution:', high_range)
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)

data.plot.scatter(x = 'GrLivArea', y = 'SalePrice', ylim = (0, 800000))
# find the Id

train.sort_values(by = 'GrLivArea',ascending = False)[:2]
# delete the two points by finding their Id

train = train.drop(train[train['Id'] ==1299].index)

train = train.drop(train[train['Id'] ==524].index)
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis = 1)

data.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice', ylim = (0, 800000))
train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice'],  fit = norm)

fig = plt.figure()

res  = stats.probplot(train['SalePrice'], plot = plt)
sns.distplot(train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot = plt)
train['GrLivArea'] = np.log(train['GrLivArea'])

sns.distplot(train['GrLivArea'], fit = norm)

fit = plt.figure()

res = stats.probplot(train['GrLivArea'], plot = plt)
sns.distplot(train['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot = plt)
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index = train.index)

train['HasBsmt'] = 0

train.loc[train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot = plt)
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.scatter(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], train[train['TotalBsmtSF'] > 0]['SalePrice'])
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x1 = train[cols].values

y1 = train['SalePrice'].values

# train the data

model = linear_model.LinearRegression()

model.fit(x1, y1)

print('The coefficients:', model.coef_)

print('The intercept:', model.intercept_)
test = pd.read_csv('../input/train.csv')

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x2 = test[cols].values

y2 = model.predict(x2)

Y = pd.DataFrame({'y2':model.predict(x2)}, index = np.arange(2, len(y2)+2))

Y = Y.drop([1461])

Y
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['SalePrice'] = np.log(sample_submission['SalePrice'])

print(mean_absolute_error(sample_submission['SalePrice'], Y))