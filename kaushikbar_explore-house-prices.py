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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
# Detect outliers using IQR method
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
df = pd.read_csv("housing.csv")
df.head()
# Summarize sales price into descriptive statistics
df['SalePrice'].describe()
# Median
df['SalePrice'].median()
# MAD 
df['SalePrice'].mad()
# COV (Coefficient of variation = ratio of standard deviation to mean)
stats.variation(df['SalePrice'])
# Skewness and Kurtosis
print("Skewness: %f" % df['SalePrice'].skew()) # Normal distribution has a skewness close to 0
print("Kurtosis: %f" % df['SalePrice'].kurt()) # Normal distribution has a kurtosis close to 3
sns.boxplot(df['SalePrice']);
# Histogram using pandas
df['SalePrice'].hist();
# Histogram using matplotlib
plt.hist(df['SalePrice']);
# Histogram with distribution plot using seaborn
sns.distplot(df['SalePrice'], fit=stats.norm);
stats.probplot(df['SalePrice'], plot=plt);
# Relationship with continuous variable
data = pd.concat([df['SalePrice'], df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
# Relationship with categorical variable
data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000);
# Correlation matrix
corrmat = df.corr()

# Plot only 10 variables with largest correlations between them
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values);
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols]);
# Find missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# Drop columns that have more than 0.1% missing data
df.drop((missing_data[missing_data['Percent'] > 0.1]).index, axis=1, inplace=True) 
# Drop a particular record with missing data for a particular column
df.drop(df.loc[df['Electrical'].isnull()].index, inplace=True) 
# Drop any remaining rows which have missing data
df.dropna(inplace=True) 

# Now let's check if we still have any missing data left
df.isnull().sum().max()
# Do we have any outliers in SalePrice?
print(outliers_iqr(df['SalePrice']))
# Do we have any outliers in GrLivArea?
print(outliers_iqr(df['GrLivArea']))
# Bivariate analysis of outliers in saleprice/grlivarea
data = pd.concat([df['SalePrice'], df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
# Delete two outliers
df.sort_values(by='GrLivArea', ascending=False)[:2]
df.drop(df[df['Id'] == 1299].index, inplace=True)
df.drop(df[df['Id'] == 524].index, inplace=True)
df['SalePrice'] = np.log(df['SalePrice'])
sns.distplot(df['SalePrice'], fit=stats.norm);
plt.figure()
stats.probplot(df['SalePrice'], plot=plt);

