# Inspired by

# https://www.kaggle.com/pmarcelino/house-prices-advanced-regression-techniques/comprehensive-data-exploration-with-python



# Load libraries

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Load data

df = pd.read_csv('../input/train.csv')

missing_values = df.isnull()



# Take logs of certain numerical variables

log_columns = ['SalePrice', 'GrLivArea', '1stFlrSF']

df[log_columns] = np.log(df[log_columns])

df.rename(columns={x: 'log' + x for x in log_columns}, inplace=True)



# Convert rating type category variables to numeric

rating_columns = df.columns[df.isin(['Ex', 'TA', 'Fa', 'Po']).any()]

f = lambda x: {'Ex': 2, 'Gd': 1, 'TA': 0, 'Fa': -1, 'Po': -2, np.nan: 0}[x]

df[rating_columns] = df[rating_columns].applymap(f)



numeric_columns = df.columns[(df.dtypes == int) | (df.dtypes == float)]

print(len(numeric_columns), 'numeric columns:', ', '.join(numeric_columns), '\n')



category_columns = df.columns[(df.dtypes == object)]

print(len(category_columns), 'category columns:', ', '.join(category_columns))
# Print columns with missing values

missing_values.sum()[missing_values.any()]
# Top 20 correlated numeric variables

df[numeric_columns].corrwith(df['logSalePrice']).sort_values()[:20:-1]
# Top 10 correlated category variables

pd.get_dummies(df[category_columns]).corrwith(df['logSalePrice']).sort_values()[:-10:-1]
# Top 10 anti-correlated category variables

pd.get_dummies(df[category_columns]).corrwith(df['logSalePrice']).sort_values()[:10]
ind = df.corr().sort_values(by='logSalePrice').index[::-1]

plt.figure(figsize=(12,10))

sns.heatmap(df[ind].corr());
sns.boxplot('OverallQual', 'logSalePrice', data=df);
sns.jointplot('logGrLivArea', 'logSalePrice', df, alpha=0.2);
sns.boxplot('GarageCars', 'logSalePrice', data=df);
sns.lmplot('GarageArea', 'logSalePrice', df, 'GarageCars', fit_reg=False);
sns.boxplot('ExterQual', 'logSalePrice', data=df);
x = df['log1stFlrSF']

y = df['logSalePrice']

cond = df['2ndFlrSF'] > 0

graph = sns.jointplot(x[~cond], y[~cond], alpha=0.2);

graph.x = x[cond]

graph.y = y[cond]

graph.plot_joint(plt.scatter, color='r', alpha=0.1)
sns.boxplot('FullBath', 'logSalePrice', data=df);
plt.subplots(figsize=(16, 8))

sns.boxplot('YearBuilt', 'logSalePrice', data=df);

plt.xticks(rotation=90);

plt.ylim(10.5,13.5);
sns.distplot(df['YearBuilt'], kde=False, bins=100);
x = 2**(np.floor(np.log2(2011 - df['YearBuilt'])) - 1)

y = df['logSalePrice']

sns.boxplot(x, y);
sns.boxplot('YrSold', 'logSalePrice', data=df);
sns.boxplot(df['MoSold'], df['logSalePrice']);
sns.boxplot('Neighborhood', 'logSalePrice', data=df)

plt.xticks(rotation=90);
sns.boxplot('GarageFinish', 'logSalePrice', data=df);
sns.boxplot('MasVnrType', 'logSalePrice', data=df);
sns.boxplot('GarageType', 'logSalePrice', data=df);
sns.boxplot('Foundation', 'logSalePrice', data=df);
sns.boxplot('SaleType', 'logSalePrice', data=df);