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
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.describe())

#ax = sns.regplot(x='OverallQual', y='SalePrice', data=train_df)
#train_df['OverallQual'].corr(train_df['SalePrice'])

#corrmat = train_df.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
#k = 10 #number of variables for heatmap
#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#cm = np.corrcoef(train_df[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()

# scatter plot
#sns.set()
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(train_df[cols], size = 2.5)
#plt.show();

#missing data
#total = train_df.isnull().sum().sort_values(ascending=False)
#percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data.head(20)

#dealing with missing data
#train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
#train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)

# outlier mgmt
# low range of SalePrice is pretty normal
# high range of SalePrice gets crazy - up to 7 stddev out
#saleprice_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:,np.newaxis]);
#low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
#high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
#print('outer range (low) of the distribution:')
#print(low_range)
#print('\nouter range (high) of the distribution:')
#print(high_range)

# Delete two outliers w/r/t GrLivArea - largest properties, but don't follow trend
#train_df.sort_values(by = 'GrLivArea', ascending = False)[:2]
#train_df = train_df.drop(train_df[train_df['Id'] == 1299].index)
#train_df = train_df.drop(train_df[train_df['Id'] == 524].index)

#deleting points
#train_df.sort_values(by = 'GrLivArea', ascending = False)[:2]
#train_df = train_df.drop(train_df[train_df['Id'] == 1299].index)
#train_df = train_df.drop(train_df[train_df['Id'] == 524].index)

# Ok, now getting to nitty gritty analysis of SalePrice
# First, test for normality
# If no normality exists, adjust using log
#applying log transformation


#transformed histogram and normal probability plot
#sns.distplot(train_df['SalePrice'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(train_df['SalePrice'], plot=plt)

#histogram and normal probability plot
#sns.distplot(train_df['GrLivArea'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(train_df['GrLivArea'], plot=plt)

# many houses don't have a basement - can't do log correction on value of 0
# so, creating variable indicating if there's a basement
# then, applying log transformation on non-zero values
train_df['HasBsmt'] = pd.Series(len(train_df['TotalBsmtSF']), index=train_df.index)
train_df['HasBsmt'] = 0 
train_df.loc[train_df['TotalBsmtSF']>0,'HasBsmt'] = 1

# log transformation of non-zero values


# Variables to predict SalePrice:
# OverallQual
# GrLivArea
# GarageArea
# TotalBsmtSF

# Take average Z score of four variables, apply that z score to prediction

#train_df['SalePrice'] = np.log(train_df['SalePrice'])
train_df['GrLivArea'] = np.log(train_df['GrLivArea'])
train_df.loc[train_df['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_df['TotalBsmtSF'])

train_df['OverallQualZScore'] = pd.Series(len(train_df['OverallQual']), index = train_df.index)
train_df['OverallQualZScore'] = (train_df['OverallQual'] - train_df['OverallQual'].mean()) / train_df['OverallQual'].std(ddof=0)

train_df['GrLivAreaZScore'] = pd.Series(len(train_df['GrLivArea']), index = train_df.index)
train_df['GrLivAreaZScore'] = (train_df['GrLivArea'] - train_df['GrLivArea'].mean()) / train_df['GrLivArea'].std(ddof=0)

train_df['TotalBsmtSFZScore'] = pd.Series(len(train_df['TotalBsmtSF']), index = train_df.index)
train_df['TotalBsmtSFZScore'] = (train_df['TotalBsmtSF'] - train_df['TotalBsmtSF'].mean()) / train_df['TotalBsmtSF'].std(ddof=0)

train_df['GarageAreaZScore'] = pd.Series(len(train_df['GarageArea']), index = train_df.index)
train_df['GarageAreaZScore'] = (train_df['GarageArea'] - train_df['GarageArea'].mean()) / train_df['GarageArea'].std(ddof=0)

train_df['AvgZScore'] = pd.Series(len(train_df['GarageAreaZScore']), index = train_df.index)
train_df['AvgZScore'] = (train_df['OverallQualZScore'] + train_df['GrLivAreaZScore'] + train_df['TotalBsmtSFZScore'] + train_df['GarageAreaZScore']) / 4

train_df['AvgZScore'].corr(train_df['SalePrice'])

train_df['Prediction'] = pd.Series(len(train_df['AvgZScore']), index = train_df.index)
train_df['Prediction'] = train_df['AvgZScore'] * train_df['SalePrice'].std(ddof=0) + train_df['SalePrice'].mean()

prediction = train_df['Prediction']
actual = train_df['SalePrice']

#mae = 29,588
mae = mean_absolute_error(actual,prediction)
print (mae)

submission = pd.DataFrame({
        "Id": train_df['Id'],
        "SalePrice": train_df['Prediction']
    })

submission.to_csv('jsm_submission_housing_prices.csv', index=False)