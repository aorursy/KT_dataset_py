# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
test_df = pd.read_csv('../input/test.csv')

test_df.head()
from scipy import stats

import seaborn as sns
train_df = pd.read_csv('../input/train.csv')

train_df.head()
train_df.count
train_df.shape[0]
print(train_df.shape[1])

train_df.columns
test_df.isnull().sum()
null_val = pd.concat([train_df.isnull().sum(),train_df.isnull().sum()/train_df.shape[0],

                     test_df.isnull().sum(),train_df.isnull().sum()/test_df.shape[0]],

                    axis =1 ,keys = ['train_null_count','%_null','test_null_count','%_null'])

null_val[null_val.sum(axis=1)>0]
train_df['SalePrice'].describe()
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)

plt.figure()

plt.subplot(1, 2, 1)

sns.distplot(train_df['SalePrice'], fit=stats.norm)

plt.subplot(1, 2, 2)

stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()

print("Skewness: %f" % train_df['SalePrice'].skew())

print("Kurtosis: %f" % train_df['SalePrice'].kurt())
plt.figure()

plt.subplot(1, 2, 1)

sns.distplot(np.log(train_df['SalePrice']+1), fit=stats.norm)

plt.subplot(1, 2, 2)

stats.probplot(np.log(train_df['SalePrice']+1), plot=plt)

plt.show()

print("Skewness: %f" % np.log(train_df['SalePrice']+1).skew())

print("Kurtosis: %f" % np.log(train_df['SalePrice']+1).kurt())
corrmat = train_df.corr()

corrmat
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cols
cm = np.corrcoef(train_df[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,

                 xticklabels=cols.values)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[cols], size=2.5)

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(1, 3)

data_total = pd.concat([train_df['SalePrice'], train_df['TotalBsmtSF']], axis=1)

data_total.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000), ax=ax1)

data1 = pd.concat([train_df['SalePrice'], train_df['1stFlrSF']], axis=1)

data1.plot.scatter(x='1stFlrSF', y='SalePrice', ylim=(0, 800000), ax=ax2)

data2 = pd.concat([train_df['SalePrice'], train_df['2ndFlrSF']], axis=1)

data2.plot.scatter(x='2ndFlrSF', y='SalePrice', ylim=(0, 800000), ax=ax3)

plt.show()
train_df['TotalBsmtSF'] = train_df['TotalBsmtSF'].fillna(0)

train_df['1stFlrSF'] = train_df['1stFlrSF'].fillna(0)

train_df['2ndFlrSF'] = train_df['2ndFlrSF'].fillna(0)

train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
data2 = pd.concat([train_df['SalePrice'], train_df['TotalSF']], axis=1)

data2.plot.scatter(x='TotalSF', y='SalePrice', ylim=(0, 800000))

plt.show()
data2 = pd.concat([train_df['SalePrice'], train_df['TotalSF']], axis=1)

data2.plot.box(x='TotalSF', y='SalePrice', ylim=(0, 800000))

plt.show()
corrmat = train_df.corr()

cols = corrmat.nsmallest(10,'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,

                 xticklabels=cols.values)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
corrmat = train_df.corr()

cols = corrmat.nsmallest(20,'SalePrice')['SalePrice'].index

cols
corrmat = train_df.corr()

corrmat.nsmallest(10,'SalePrice')['SalePrice']
corrmat = train_df.corr().abs()

corrmat.nsmallest(10,'SalePrice')
corrmat = train_df.corr().abs()

cols = corrmat.nsmallest(10,'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,

                 xticklabels=cols.values)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
corrmat = train_df.corr().abs()

corrmat.nsmallest(30,'SalePrice')['SalePrice']
categoricals = train_df.select_dtypes(exclude=[np.number])

categoricals.describe()
categoricals.isnull().sum()
category_null = pd.concat([categoricals.isnull().sum(),categoricals.isnull().sum()/categoricals.shape[0]],axis =1,keys=['category_count','%count'])

category_null[category_null.sum(axis=1)>0]
categoricals.columns
train_df[ 'RoofMatl'].describe()
roof = pd.get_dummies(train_df.RoofMatl, drop_first=True)

roof.describe()
df = pd.concat([roof,train_df['SalePrice']],axis=1)
df.head()
corrmat = df.corr()

cols = corrmat.nsmallest(8,'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,

                 xticklabels=cols.values)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
train_df['OverallQual'].describe()
f, ax = plt.subplots(figsize=(8, 6))

fig = sns.distplot(train_df['OverallQual'])

plt.show()
overall_qual = pd.concat([train_df['SalePrice'], train_df['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=overall_qual)

fig.axis(ymin=0, ymax=800000)

plt.show()
overall_qual.plot.scatter(x='OverallQual', y="SalePrice")

plt.show()
data = pd.concat([train_df['SalePrice'],train_df['LotFrontage']],axis=1)

data.plot.scatter(x='LotFrontage', y="SalePrice")

plt.show()
train_df['LotConfig'].describe()
data = pd.concat([train_df['SalePrice'],train_df['LotConfig']],axis=1)

fig = sns.boxplot(x='LotConfig', y="SalePrice", data=data)

plt.show()
train_df['LotShape'].describe()
data = pd.concat([train_df['SalePrice'],train_df['LotShape']],axis=1)

fig = sns.boxplot(x='LotShape', y="SalePrice", data=data)

plt.show()
data = pd.concat([train_df['SalePrice'],train_df['LotArea']],axis=1)

data.plot.scatter(x='LotArea', y="SalePrice")

plt.show()
train_df['LandContour'].describe()
data = pd.concat([train_df['LandContour'],train_df['SalePrice']],axis=1)

fig = sns.boxplot(x='LandContour',y='SalePrice',data=data)

plt.show()
train_df['LotShape'].describe()
data = pd.concat([train_df['SalePrice'],train_df['LotShape']],axis=1)

fig = sns.boxplot(x='LotShape', y="SalePrice",data=data)

plt.show()
train_df['Condition1'].describe()
data = pd.concat([train_df['SalePrice'],train_df['Condition1']],axis=1)

fig = sns.boxplot(x='Condition1', y="SalePrice",data=data)

plt.show()
train_df['Condition2'].describe()
data = pd.concat([train_df['SalePrice'],train_df['Condition2']],axis=1)

fig = sns.boxplot(x='Condition2', y="SalePrice",data=data)

plt.show()
train_df['Neighborhood'].describe()
data = pd.concat([train_df['SalePrice'],train_df['Neighborhood']],axis=1)

fig = sns.boxplot(x='Neighborhood', y="SalePrice",data=data)

plt.show()
train_df['OverallCond'].describe()
data = pd.concat([train_df['SalePrice'],train_df['OverallCond']],axis=1)

fig = sns.boxplot(x='OverallCond', y="SalePrice",data=data)

plt.show()
data.plot.scatter(x='OverallCond', y="SalePrice")

plt.show()
data = pd.concat([train_df['SalePrice'],train_df['TotalSF']],axis=1)

data.plot.scatter(x='TotalSF', y="SalePrice")

plt.show()
plt.figure();

plt.subplot(1,2,1)

sns.distplot(train_df['TotalSF'],fit=stats.norm)

plt.subplot(1,2,2)

stats.probplot(train_df['TotalSF'],plot=plt)

plt.show()
print(train_df['TotalSF'].skew())

print(train_df['TotalSF'].kurt())
plt.figure();

plt.subplot(1,2,1)

sns.distplot(np.log(train_df['TotalSF']),fit=stats.norm)

plt.subplot(1,2,2)

stats.probplot(np.log(train_df['TotalSF']),plot=plt)

plt.show()
print(np.log(train_df['TotalSF'].skew()))
print(np.log(train_df['TotalSF'].kurt()))
plt.figure();

plt.subplot(1,2,1)

sns.distplot(train_df['GrLivArea'],fit=stats.norm)

plt.subplot(1,2,2)

stats.probplot(train_df['GrLivArea'],plot=plt)

plt.show()

print(train_df['GrLivArea'].skew())

print(train_df['GrLivArea'].kurt())
plt.figure();

plt.subplot(1,2,1)

sns.distplot(np.log(train_df['GrLivArea']),fit=stats.norm)

plt.subplot(1,2,2)

stats.probplot(np.log(train_df['GrLivArea']),plot=plt)

plt.show()

print(np.log(train_df['GrLivArea']).skew())

print(np.log(train_df['GrLivArea']).kurt())
train_df['TotalBsmtSF'].describe()
data1 = pd.concat([train_df['TotalBsmtSF'],train_df['SalePrice']],axis=1)

data1.plot.scatter(x='TotalBsmtSF',y='SalePrice')

plt.show()
f, ax1 = plt.subplots(figsize=(20,15) ,ncols=2, nrows=2)



left   =  0.125  # the left side of the subplots of the figure

right  =  0.7    # the right side of the subplots of the figure

bottom =  0.1    # the bottom of the subplots of the figure

top    =  0.5    # the top of the subplots of the figure

wspace =  .5     # the amount of width reserved for blank space between subplots

hspace =  1.1    # the amount of height reserved for white space between subplots





plt.subplots_adjust(

    left    =  left, 

    bottom  =  bottom, 

    right   =  right, 

    top     =  top, 

    wspace  =  wspace, 

    hspace  =  hspace

)

data1 = pd.concat([train_df['TotalBsmtSF'],train_df['SalePrice']],axis=1)

data1.plot.scatter(x='TotalBsmtSF',y='SalePrice',ax=ax1[0][0])

data1 = pd.concat([train_df['BsmtFinSF1'],train_df['SalePrice']],axis=1)

data1.plot.scatter(x='BsmtFinSF1',y='SalePrice',ax=ax1[0][1])

data1 = pd.concat([train_df['BsmtFinSF2'],train_df['SalePrice']],axis=1)

data1.plot.scatter(x='BsmtFinSF2',y='SalePrice',ax=ax1[1][0])

data1 = pd.concat([train_df['BsmtUnfSF'],train_df['SalePrice']],axis=1)

data1.plot.scatter(x='BsmtUnfSF',y='SalePrice',ax=ax1[1][1])

plt.show()
train_df['Exterior2nd'].describe()
train_df['Exterior2nd'].mode()
train_df['MSSubClass'].describe()
train_df['MSSubClass'].head()
train_df1 = pd.get_dummies(train_df,drop_first=True)
train_df1.head()
## After this we do variable dropping and variable manipulation for linear regression

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.drop(['Id','Alley','FireplaceQu', 'PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
train.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1, inplace=True)
train['TotalBsmtSF'] = train['TotalBsmtSF'].fillna(0)

train['1stFlrSF'] = train['1stFlrSF'].fillna(0)

train['2ndFlrSF'] = train['2ndFlrSF'].fillna(0)

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

train.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

train.drop(['GarageArea','TotRmsAbvGrd'], axis=1, inplace=True) # as analysis before
train.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'MasVnrType', 'Heating', 'LowQualFinSF',

            'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

            'Functional', 'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'WoodDeckSF',

            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

            'MiscVal'], axis=1, inplace=True)
## Normalize data 

numeric_data = train.loc[:, ['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]

numeric_data_standardized = (numeric_data - numeric_data.mean())/numeric_data.std()
#  Fillling nan values 

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])

train['GarageCars'] = train['GarageCars'].fillna(0.0)

train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])

train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior2nd'].mode()[0])

train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])


train['MSSubClass'] = train['MSSubClass'].astype(str)

train['OverallCond'] = train['OverallCond'].astype(str)

train['KitchenAbvGr'] = train['KitchenAbvGr'].astype(str)



train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)

def encode(x): return 1 if x == 'Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode)

train.drop(['SaleType'],axis=1,inplace=True)
train.isnull().sum()
sum(train.isnull().sum() != 0)
print(train.shape[0])

print(train.shape[1])
train.head()
y = np.log(train.SalePrice)

X = train
X.drop(['SalePrice'],axis=1,inplace=True)
print(y.shape[0])

print(X.shape[0])

print(X.shape[1])
X.dtypes.sample(20)
X1_hot_code = pd.get_dummies(X,drop_first=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                                    X1_hot_code, y, random_state=42, test_size=.33)

from sklearn import linear_model

lr = linear_model.LinearRegression()



model = lr.fit(X_train, y_train)
print ("R^2 is: n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print ('RMSE is: n', mean_squared_error(y_test, predictions))
actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.75,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
for i in range (-2, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(X_train, y_train)

    preds_ridge = ridge_model.predict(X_test)

    

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')

    plt.xlabel('Predicted Price')

    plt.ylabel('Actual Price')

    plt.title('Ridge Regularization with alpha = {}'.format(alpha))

    overlay = 'R^2 is: {}nRMSE is: {}'.format(

                    ridge_model.score(X_test, y_test),

                    mean_squared_error(y_test, preds_ridge))

    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

    plt.show()