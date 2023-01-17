import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_new = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df = df.drop(columns='Id')

df_new = df_new.drop(columns='Id')
print(df.info())

print('\n',df_new.info())
print('MSSubClass Unique Values:')

print(df.MSSubClass.unique())

print(df_new.MSSubClass.unique())

print('\nYrSold Unique Values')

print(df.YrSold.unique())

print(df_new.YrSold.unique())
df.MSSubClass = df.MSSubClass.astype(object)

df_new.MSSubClass = df_new.MSSubClass.astype(object)



df.YrSold = df.YrSold.astype(object)

df_new.YrSold = df_new.YrSold.astype(object)
from sklearn.impute import SimpleImputer
bsmt_df = df[df.columns[df.columns.str.contains('Bsmt')]]

print(bsmt_df.head())

print(bsmt_df.isnull().sum())
no_bsmt_df = bsmt_df[bsmt_df.TotalBsmtSF == 0]

with_bsmt_df = bsmt_df[bsmt_df.TotalBsmtSF > 0]
# IMPUTE NO_BASEMENT!!

cat_no_bsmt_df = no_bsmt_df.select_dtypes(object)

num_no_bsmt_df = no_bsmt_df.select_dtypes(exclude=object)



imp_no_bsmt_df = SimpleImputer(strategy='constant', fill_value='no basement')

imp_zero = SimpleImputer(strategy='constant', fill_value=0)



imp_no_bsmt_df.fit(cat_no_bsmt_df)

imp_zero.fit(num_no_bsmt_df)



cat_no_bsmt_df = pd.DataFrame(imp_no_bsmt_df.transform(cat_no_bsmt_df), index = cat_no_bsmt_df.index, columns = cat_no_bsmt_df.columns)

num_no_bsmt_df = pd.DataFrame(imp_zero.transform(num_no_bsmt_df), index = num_no_bsmt_df.index, columns = num_no_bsmt_df.columns)



no_bsmt_df = pd.concat([cat_no_bsmt_df, num_no_bsmt_df], axis=1)

print(no_bsmt_df.isnull().sum())
# IMPUTE WITH BASEMENTS!!

cat_with_bsmt_df = with_bsmt_df.select_dtypes(object)

num_with_bsmt_df = with_bsmt_df.select_dtypes(exclude=object)



imp_freq = SimpleImputer(strategy='most_frequent')

imp_mean = SimpleImputer(strategy='mean')



imp_freq.fit(cat_with_bsmt_df)

imp_mean.fit(num_with_bsmt_df)



cat_with_bsmt_df = pd.DataFrame(imp_freq.transform(cat_with_bsmt_df), index=cat_with_bsmt_df.index, columns=cat_with_bsmt_df.columns)

num_with_bsmt_df = pd.DataFrame(imp_mean.transform(num_with_bsmt_df), index=num_with_bsmt_df.index, columns=num_with_bsmt_df.columns)



with_bsmt_df = pd.concat([cat_with_bsmt_df, num_with_bsmt_df], axis=1)

print(with_bsmt_df.isnull().sum())
bsmt_df = pd.concat([with_bsmt_df, no_bsmt_df])

bsmt_df.sort_index(inplace=True)

bsmt_df.head()
bsmt_df_new = df_new[df_new.columns[df_new.columns.str.contains('Bsmt')]]

print(bsmt_df_new.head())

print(bsmt_df_new.isnull().sum())
no_bsmt_df_new = bsmt_df_new[(bsmt_df_new.TotalBsmtSF == 0) | (bsmt_df_new.TotalBsmtSF.isnull() == True)]

with_bsmt_df_new = bsmt_df_new[bsmt_df_new.TotalBsmtSF > 0]
# IMPUTE NO_BASEMENT!!

cat_no_bsmt_df_new = no_bsmt_df_new.select_dtypes(object)

num_no_bsmt_df_new = no_bsmt_df_new.select_dtypes(exclude=object)



imp_no_bsmt_df_new = SimpleImputer(strategy='constant', fill_value='no basement')

imp_zero = SimpleImputer(strategy='constant', fill_value=0)



imp_no_bsmt_df_new.fit(cat_no_bsmt_df_new)

imp_zero.fit(num_no_bsmt_df_new)



cat_no_bsmt_df_new = pd.DataFrame(imp_no_bsmt_df_new.transform(cat_no_bsmt_df_new), index = cat_no_bsmt_df_new.index, columns = cat_no_bsmt_df_new.columns)

num_no_bsmt_df_new = pd.DataFrame(imp_zero.transform(num_no_bsmt_df_new), index = num_no_bsmt_df_new.index, columns = num_no_bsmt_df_new.columns)



no_bsmt_df_new = pd.concat([cat_no_bsmt_df_new, num_no_bsmt_df_new], axis=1)

print(no_bsmt_df_new.isnull().sum())
# IMPUTE WITH BASEMENTS!!

cat_with_bsmt_df_new = with_bsmt_df_new.select_dtypes(object)

num_with_bsmt_df_new = with_bsmt_df_new.select_dtypes(exclude=object)



imp_freq = SimpleImputer(strategy='most_frequent')

imp_mean = SimpleImputer(strategy='mean')



imp_freq.fit(cat_with_bsmt_df_new)

imp_mean.fit(num_with_bsmt_df_new)



cat_with_bsmt_df_new = pd.DataFrame(imp_freq.transform(cat_with_bsmt_df_new), index=cat_with_bsmt_df_new.index, columns=cat_with_bsmt_df_new.columns)

num_with_bsmt_df_new = pd.DataFrame(imp_mean.transform(num_with_bsmt_df_new), index=num_with_bsmt_df_new.index, columns=num_with_bsmt_df_new.columns)



with_bsmt_df_new = pd.concat([cat_with_bsmt_df_new, num_with_bsmt_df_new], axis=1)

print(with_bsmt_df_new.isnull().sum())
bsmt_df_new = pd.concat([with_bsmt_df_new, no_bsmt_df_new])

bsmt_df_new.sort_index(inplace=True)

bsmt_df_new.head()
garage_df = df[df.columns[df.columns.str.contains('Garage')]]

print(garage_df.head())

print(garage_df.isnull().sum())
no_garage_df = garage_df[(garage_df.GarageArea == 0)]

print(no_garage_df.isnull().sum())

with_garage_df = garage_df[(garage_df.GarageArea > 0)]

print('\n')

print(with_garage_df.isnull().sum())
# IMPUTE NO_GARAGE!!

cat_no_garage_df = no_garage_df.select_dtypes(object)

num_no_garage_df = no_garage_df.select_dtypes(exclude=object)



imp_no_garage_df = SimpleImputer(strategy='constant', fill_value='no basement')

imp_zero = SimpleImputer(strategy='constant', fill_value=0)



imp_no_garage_df.fit(cat_no_garage_df)

imp_zero.fit(num_no_garage_df)



cat_no_garage_df = pd.DataFrame(imp_no_garage_df.transform(cat_no_garage_df), index = cat_no_garage_df.index, columns = cat_no_garage_df.columns)

num_no_garage_df = pd.DataFrame(imp_zero.transform(num_no_garage_df), index = num_no_garage_df.index, columns = num_no_garage_df.columns)



no_garage_df = pd.concat([cat_no_garage_df, num_no_garage_df], axis=1)

print(no_garage_df.isnull().sum())
garage_df = pd.concat([no_garage_df, with_garage_df])

garage_df.sort_index(inplace=True)

garage_df.head()
garage_df_new = df_new[df_new.columns[df_new.columns.str.contains('Garage')]]

print(garage_df_new.head())

print(garage_df_new.isnull().sum())
no_garage_df_new = garage_df_new[(garage_df_new.GarageArea == 0) | (garage_df_new.GarageArea.isnull())]

print(no_garage_df_new.isnull().sum())

with_garage_df_new = garage_df_new[(garage_df_new.GarageArea > 0)]

print('\n')

print(with_garage_df_new.isnull().sum())
garage_df_new.loc[1116, 'GarageType'] = np.nan

garage_df_new.loc[1116]
# IMPUTE NO_GARAGE!!

cat_no_garage_df_new = no_garage_df_new.select_dtypes(object)

num_no_garage_df_new = no_garage_df_new.select_dtypes(exclude=object)



imp_no_garage_df_new = SimpleImputer(strategy='constant', fill_value='no basement')

imp_zero = SimpleImputer(strategy='constant', fill_value=0)



imp_no_garage_df_new.fit(cat_no_garage_df_new)

imp_zero.fit(num_no_garage_df_new)



cat_no_garage_df_new = pd.DataFrame(imp_no_garage_df_new.transform(cat_no_garage_df_new), index = cat_no_garage_df_new.index, columns = cat_no_garage_df_new.columns)

num_no_garage_df_new = pd.DataFrame(imp_zero.transform(num_no_garage_df_new), index = num_no_garage_df_new.index, columns = num_no_garage_df_new.columns)



no_garage_df_new = pd.concat([cat_no_garage_df_new, num_no_garage_df_new], axis=1)

print(no_garage_df_new.isnull().sum())
# IMPUTE WITH GARAGE!!

cat_with_garage_df_new = with_garage_df_new.select_dtypes(object)

num_with_garage_df_new = with_garage_df_new.select_dtypes(exclude=object)



imp_freq = SimpleImputer(strategy='most_frequent')

imp_mean = SimpleImputer(strategy='mean')



imp_freq.fit(cat_with_garage_df_new)

imp_mean.fit(num_with_garage_df_new)



cat_with_garage_df_new = pd.DataFrame(imp_freq.transform(cat_with_garage_df_new), index=cat_with_garage_df_new.index, columns=cat_with_garage_df_new.columns)

num_with_garage_df_new = pd.DataFrame(imp_mean.transform(num_with_garage_df_new), index=num_with_garage_df_new.index, columns=num_with_garage_df_new.columns)



with_garage_df_new = pd.concat([cat_with_garage_df_new, num_with_garage_df_new], axis=1)

print(with_garage_df_new.isnull().sum())
garage_df_new = pd.concat([no_garage_df_new, with_garage_df_new])

garage_df_new.sort_index(inplace=True)

garage_df_new.head()
pool_df = df[df.columns[df.columns.str.contains('Pool')]]

pool_df.head()

pool_df.isnull().sum()
pool_df[pool_df.PoolArea > 0]
pool_df.PoolQC = pool_df.PoolQC.fillna('no pool')

pool_df.PoolQC.unique()
pool_df_new = df_new[df_new.columns[df_new.columns.str.contains('Pool')]]

pool_df_new.head()

pool_df_new.isnull().sum()
pool_df_new[pool_df_new.PoolArea > 0]
pool_df_new.PoolQC[pool_df_new.PoolArea > 0] = pool_df_new.PoolQC[pool_df_new.PoolArea > 0].fillna('Gd')

pool_df_new[pool_df_new.PoolArea > 0]
pool_df_new.PoolQC = pool_df_new.PoolQC.fillna('no pool')

pool_df_new.PoolQC.unique()
masvnr_df = df[df.columns[df.columns.str.contains('Mas')]]

print(masvnr_df.head())

print(masvnr_df.isnull().sum())
masvnr_df[masvnr_df.MasVnrType.isnull()]
masvnr_df.MasVnrType = masvnr_df.MasVnrType.fillna('None')

masvnr_df.MasVnrArea = masvnr_df.MasVnrArea.fillna(0)
masvnr_df.MasVnrType.unique()
masvnr_df_new = df_new[df_new.columns[df_new.columns.str.contains('Mas')]]

print(masvnr_df_new.head())

print(masvnr_df_new.isnull().sum())
masvnr_df_new[masvnr_df_new.MasVnrType.isnull()]
masvnr_df_new.MasVnrArea = masvnr_df_new.MasVnrArea.fillna(0)

masvnr_df_new.MasVnrType[masvnr_df_new.MasVnrArea == 0] = masvnr_df_new.MasVnrType[masvnr_df_new.MasVnrArea == 0].fillna('None')
masvnr_df_new.MasVnrType.value_counts()
mask = (masvnr_df_new.MasVnrType.isnull() == True) & (masvnr_df_new.MasVnrArea > 0)

masvnr_df_new.MasVnrType[mask] = masvnr_df_new.MasVnrType[mask].fillna(masvnr_df_new.MasVnrType.value_counts().index[1])
print(masvnr_df_new.isnull().sum())

print(masvnr_df_new.MasVnrType.unique())
fireplace_df = df[df.columns[df.columns.str.contains('Fire')]]

print(fireplace_df.head())

print(fireplace_df.isnull().sum())
fireplace_df.FireplaceQu[fireplace_df.Fireplaces == 0] = fireplace_df.FireplaceQu[fireplace_df.Fireplaces == 0].fillna('no fireplace')
print(fireplace_df.isnull().sum())
fireplace_df_new = df_new[df_new.columns[df_new.columns.str.contains('Fire')]]

print(fireplace_df_new.head())

print(fireplace_df_new.isnull().sum())
fireplace_df_new.FireplaceQu[fireplace_df_new.Fireplaces == 0] = fireplace_df_new.FireplaceQu[fireplace_df_new.Fireplaces == 0].fillna('no fireplace')
print(fireplace_df_new.isnull().sum())
col_used = np.concatenate((garage_df.columns.values, bsmt_df.columns.values, pool_df.columns.values,

                           masvnr_df.columns.values, fireplace_df.columns.values))

remaining_df = df.drop(col_used, axis=1)

remaining_df_new = df_new.drop(col_used, axis=1)
remaining_df.isnull().sum()[remaining_df.isnull().sum() > 0]
remaining_df.LotFrontage = remaining_df.LotFrontage.fillna(remaining_df.LotFrontage.mean())

remaining_df.Alley = remaining_df.Alley.fillna('no alley')

remaining_df.Electrical = remaining_df.Electrical.fillna(remaining_df.Electrical.value_counts().index[0])

remaining_df.Fence = remaining_df.Fence.fillna('no fence')

remaining_df.MiscFeature = remaining_df.MiscFeature.fillna('no misc feature')
df = pd.concat([remaining_df, fireplace_df, pool_df, masvnr_df, garage_df, bsmt_df], axis=1)

df['test_set'] = 0

print(df.info())
remaining_df_new.isnull().sum()[remaining_df_new.isnull().sum() > 0]
remaining_df_new.LotFrontage = remaining_df_new.LotFrontage.fillna(remaining_df_new.LotFrontage.mean())

remaining_df_new.Alley = remaining_df_new.Alley.fillna('no alley')

remaining_df_new.Fence = remaining_df_new.Fence.fillna('no fence')

remaining_df_new.MiscFeature = remaining_df_new.MiscFeature.fillna('no misc feature')

remaining_df_new.Utilities = remaining_df_new.Utilities.fillna(remaining_df_new.Utilities.value_counts().index[0])

remaining_df_new.Exterior1st = remaining_df_new.Exterior1st.fillna(remaining_df_new.Exterior1st.value_counts().index[0])

remaining_df_new.Exterior2nd = remaining_df_new.Exterior2nd.fillna(remaining_df_new.Exterior2nd.value_counts().index[0])

remaining_df_new.KitchenQual = remaining_df_new.KitchenQual.fillna(remaining_df_new.KitchenQual.value_counts().index[0])

remaining_df_new.Functional = remaining_df_new.Functional.fillna(remaining_df_new.Functional.value_counts().index[0])

remaining_df_new.SaleType = remaining_df_new.SaleType.fillna(remaining_df_new.SaleType.value_counts().index[0])

remaining_df_new.MSZoning = remaining_df_new.MSZoning.fillna(remaining_df_new.MSZoning.value_counts().index[0])
df_new = pd.concat([remaining_df_new, fireplace_df_new, pool_df_new, masvnr_df_new, garage_df_new, bsmt_df_new], axis=1)

df_new['test_set'] = 1

print(df_new.info())
# Check if 'TotalBsmtSF' is the sum of BsmtFinSF1, BsmtFinSF2, and BsmtUnfSF

if (df.BsmtFinSF1 + df.BsmtFinSF2 + df.BsmtUnfSF == df.TotalBsmtSF).sum() == len(df):

    df = df.drop(columns='TotalBsmtSF')

if (df_new.BsmtFinSF1 + df_new.BsmtFinSF2 + df_new.BsmtUnfSF == df_new.TotalBsmtSF).sum() == len(df_new):

    df_new = df_new.drop(columns='TotalBsmtSF')

print("'TotalBsmtSF' dropped")
# Encode Categorical Features

data = pd.concat([df, df_new], ignore_index=True)

data = pd.get_dummies(data)



# Set Training & Test Features

X = data[data.test_set == 0].drop(columns=['test_set', 'SalePrice'])

X_pred = data[data.test_set == 1].drop(columns=['test_set', 'SalePrice'])

print('Training features shape:')

print(X.shape)

print('\nTest features shape:')

print(X_pred.shape)
y = df.SalePrice

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.hist(y)

ax2.hist(np.log(y))
# Set Target (Logarithmic Transformation)

y = np.log(df.SalePrice)

print('Target shape:')

print(y.shape)
# Validation Split (30%)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

param_gbr = {'n_estimators' : np.arange(500, 1100, 100)}

gbr = GradientBoostingRegressor()

gbr_cv = GridSearchCV(gbr, param_gbr, cv=3)

gbr_cv.fit(X_train, y_train)

print('GBR Best Params & Score:')

print(gbr_cv.best_params_)

print(gbr_cv.best_score_)
from sklearn.metrics import mean_squared_log_error

y_val = np.exp(gbr_cv.predict(X_test))

y_true = np.exp(y_test)

print('Validation RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(y_true, y_val))))
# Re-train GB Regressor

gbr_cv.fit(X, y)

print('GBR Best Params & Score:')

print(gbr_cv.best_params_)

print(gbr_cv.best_score_)
y_pred = np.exp(gbr_cv.predict(X_pred))

print('Predictions Shape:')

print(y.shape)
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col=0)

print(submission.head())

submission.SalePrice = y_pred

print('\n')

print(submission.head())
submission.to_csv('submission.csv')

print('Result saved successfully!')