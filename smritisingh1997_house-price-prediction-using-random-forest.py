import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from scipy import stats

import warnings

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_columns = None

pd.options.display.max_rows = None
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(df_train['SalePrice'])
print('Skewness: {}'.format(df_train.SalePrice.skew()))

print('Kurtosis: {}'.format(df_train.SalePrice.kurt()))
corrmat = df_train.corr()
k = 12

cols = corrmat.nlargest(k, 'SalePrice') ['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

fig, ax = plt.subplots(figsize=(10,10))

sns.set(font_scale = 1)

hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 12}, 

                 yticklabels = cols.values, xticklabels = cols.values)

plt.show()
cor_target = abs(corrmat["SalePrice"])

relevant_features = cor_target[cor_target>0.3]

relevant_features

# corrmat
# def check_corr(s):

#     print(df_train[['YearBuilt', s]].corr())

    

# for i in cols_train:

#     check_corr(i)
# print(df_train[['OverallQual', 'LotFrontage']].corr())

# print(df_train[['OverallQual', 'YearBuilt']].corr())

# print(df_train[['OverallQual','YearRemodAdd']].corr())

# print(df_train[['OverallQual','MasVnrArea']].corr())

# print(df_train[['OverallQual','BsmtFinSF1']].corr())

# print(df_train[['OverallQual','TotalBsmtSF']].corr())

# print(df_train[['OverallQual','1stFlrSF']].corr())

# print(df_train[['OverallQual','2ndFlrSF']].corr())

# print(df_train[['OverallQual','GrLivArea']].corr())

# print(df_train[['OverallQual','FullBath']].corr())

# print(df_train[['OverallQual','TotRmsAbvGrd']].corr())

# print(df_train[['OverallQual','Fireplaces']].corr())

# print(df_train[['OverallQual','GarageYrBlt']].corr())

# print(df_train[['OverallQual','GarageCars']].corr())

# print(df_train[['OverallQual','GarageArea']].corr())

# print(df_train[['OverallQual','WoodDeckSF']].corr())

# print(df_train[['OverallQual','OpenPorchSF']].corr())
cols_train = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',

                          '1stFlrSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',

                          'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF',

                          'OpenPorchSF']

cols_test = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',

                          '1stFlrSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',

                          'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF',

                          'OpenPorchSF']
df_train_final = df_train[cols_train]

df_test_final = df_test[cols_test]
sns.set()

sns.pairplot(df_train_final[cols_train], size=2.5)

plt.show()
total = df_train_final.isnull().sum().sort_values(ascending = False)

percent = (df_train_final.isnull().sum() / df_train_final.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis=1, keys = ['Total', 'Percent'])

# missing_data
total = df_test_final.isnull().sum().sort_values(ascending = False)

percent = (df_test_final.isnull().sum() / df_test_final.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis=1, keys = ['Total', 'Percent'])
df_train_final = df_train_final.drop((missing_data[missing_data['Total'] > 7]).index,1)

df_train_final.isnull().sum().max()
df_test_final = df_test_final.drop((missing_data[missing_data['Total'] > 14]).index,1)

df_test_final = df_test_final.fillna(0)

df_test_final.isnull().sum().max()
sc = StandardScaler()

# sc1 = StandardScaler()

saleprice_scaled = sc.fit_transform(df_train_final['SalePrice'][:,np.newaxis])

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
var = 'GrLivArea'

data = pd.concat([df_train_final['SalePrice'], df_train_final[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
# df_train_final.sort_values(by = 'GrLivArea', ascending=False)[:2]

# df_train_final = df_train_final.drop([1298, 523])

# df_test_final.sort_values(by = 'GrLivArea', ascending=False)[:2]

# df_test_final = df_test_final.drop([1089, 728])
var = 'TotalBsmtSF'

data = pd.concat([df_train_final['SalePrice'], df_train_final[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
var = 'OverallQual'

data = pd.concat([df_train_final['SalePrice'], df_train_final[var]], axis = 1)

f, ax = plt.subplots(figsize = (8, 5))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ymin = 0, ymax = 800000)
var = 'YearBuilt'

data = pd.concat([df_train_final['SalePrice'], df_train_final[var]], axis = 1)

f, ax = plt.subplots(figsize = (8, 5))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ymin = 0, ymax = 800000)
sns.distplot(df_train_final['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train_final['SalePrice'], plot = plt)
sns.distplot(df_train_final['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train_final['GrLivArea'], plot = plt)
df_train_final['GrLivArea'] = np.log(df_train_final['GrLivArea'])

df_test_final['GrLivArea'] = np.log(df_test_final['GrLivArea'])
sns.distplot(df_train_final['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train_final['GrLivArea'], plot = plt)
sns.distplot(df_train_final['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train_final['TotalBsmtSF'], plot = plt)
df_train_final['HasBsmt'] = pd.Series(len(df_train_final['TotalBsmtSF']), index = df_train.index)

df_test_final['HasBsmt'] = pd.Series(len(df_test_final['TotalBsmtSF']), index = df_test.index)

df_train_final['HasBsmt'] = 0

df_test_final['HasBsmt'] = 0

df_train_final.loc[df_train_final['TotalBsmtSF']>0,'HasBsmt'] = 1

df_test_final.loc[df_test_final['TotalBsmtSF']>0,'HasBsmt'] = 1
df_train_final.loc[df_train_final['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train_final['TotalBsmtSF'])

df_test_final.loc[df_test_final['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_test_final['TotalBsmtSF'])
sns.distplot(df_train_final[df_train_final['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train_final[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot = plt)
plt.scatter(df_train_final['GrLivArea'], df_train_final['SalePrice'])
plt.scatter(df_train_final[df_train_final['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train_final[df_train_final['TotalBsmtSF'] > 0]['SalePrice'])
cols_train_final = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',

                          '1stFlrSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd'

                          , 'Fireplaces', 'BsmtFinSF1', 'WoodDeckSF', '2ndFlrSF',

                          'OpenPorchSF']

cols_test_final = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',

                          '1stFlrSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd'

                          , 'Fireplaces', 'BsmtFinSF1', 'WoodDeckSF', '2ndFlrSF',

                          'OpenPorchSF']
df_train_final[cols_train_final].head()
df_train_final = df_train_final[cols_train_final]

df_test_final = df_test_final[cols_test_final]
df_train_final = df_train_final[cols_train_final]

df_test_final = df_test_final[cols_test_final]
# sc.fit(df_train_final)

# sc1.fit(df_test_final)

# tranformed_df = pd.DataFrame(df_train_final_scaled_X, columns=cols_train_final)

# tranformed_df_test = pd.DataFrame(df_test_final_scaled_X, columns=cols_train_final)

# final_train_data = tranformed_df.copy()

# final_test_data = tranformed_df_test.copy()

# final_train_data[['GrLivArea', 'TotalBsmtSF', 'SalePrice']] = df_train_final[['GrLivArea', 'TotalBsmtSF', 'SalePrice']]

# final_test_data[['GrLivArea', 'TotalBsmtSF']] = df_test_final[['GrLivArea', 'TotalBsmtSF']]

# final_train_data.head()
final_train_data_dummies = pd.get_dummies(df_train_final)

final_test_data_dummies = pd.get_dummies(df_test_final)
rf = RandomForestRegressor(n_estimators=20)
X = final_train_data_dummies.loc[:,final_train_data_dummies.columns != 'SalePrice']

y = final_train_data_dummies.SalePrice
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf_fit = rf.fit(X,y)
y = rf_fit.predict(final_test_data_dummies)
submission_df = pd.DataFrame(y,columns=['SalePrice'])
submission_df['Id'] = df_test['Id']

submission_df = submission_df[['Id', 'SalePrice']]
submission_df.to_csv('/kaggle/working/submission.csv', index=False)