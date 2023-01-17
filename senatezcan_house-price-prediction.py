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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(60, 40))

sns.heatmap(corrmat, vmax=.8, square=True,annot=True);


k = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)

missing_train_data = pd.concat([total], axis=1, keys=['Total'])

missing_train_data 
df_train["PoolQC"] = df_train["PoolQC"].fillna("None")



df_train["MiscFeature"] = df_train["MiscFeature"].fillna("None")



df_train["Alley"] = df_train["Alley"].fillna("None")



df_train["Fence"] = df_train["Fence"].fillna("None")



df_train["FireplaceQu"] = df_train["FireplaceQu"].fillna("None")





df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df_train[col] = df_train[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df_train[col] = df_train[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_train[col] = df_train[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_train[col] = df_train[col].fillna('None')



df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")

df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)



df_train['MSZoning'] = df_train['MSZoning'].fillna(df_train['MSZoning'].mode()[0])



df_train = df_train.drop(['Utilities'], axis=1)



df_train["Functional"] = df_train["Functional"].fillna("Typ")



df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])



df_train['KitchenQual'] = df_train['KitchenQual'].fillna(df_train['KitchenQual'].mode()[0])



df_train['Exterior1st'] = df_train['Exterior1st'].fillna(df_train['Exterior1st'].mode()[0])

df_train['Exterior2nd'] = df_train['Exterior2nd'].fillna(df_train['Exterior2nd'].mode()[0])



df_train['SaleType'] = df_train['SaleType'].fillna(df_train['SaleType'].mode()[0])



df_train['MSSubClass'] = df_train['MSSubClass'].fillna("None")
total = df_test.isnull().sum().sort_values(ascending=False)

missing_test_data = pd.concat([total], axis=1, keys=['Total'])

missing_test_data
df_test["PoolQC"] = df_test["PoolQC"].fillna("None")



df_test["MiscFeature"] = df_test["MiscFeature"].fillna("None")



df_test["Alley"] = df_test["Alley"].fillna("None")



df_test["Fence"] = df_test["Fence"].fillna("None")



df_test["FireplaceQu"] = df_test["FireplaceQu"].fillna("None")





df_test["LotFrontage"] = df_test.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df_test[col] = df_test[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df_test[col] = df_test[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_test[col] = df_test[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_test[col] = df_test[col].fillna('None')



df_test["MasVnrType"] = df_test["MasVnrType"].fillna("None")

df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(0)



df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])



df_test = df_test.drop(['Utilities'], axis=1)



df_test["Functional"] = df_test["Functional"].fillna("Typ")



df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])



df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])



df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])

df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])



df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])



df_test['MSSubClass'] = df_test['MSSubClass'].fillna("None")
df_train = pd.DataFrame(df_train)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

df_train.describe()

df_train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

df_train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
df_train['YrBltAndRemod']=df_train['YearBuilt']+df_train['YearRemodAdd']

df_train['TotalSF']=df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']



df_train['Total_sqr_footage'] = (df_train['BsmtFinSF1'] + df_train['BsmtFinSF2'] +

                                 df_train['1stFlrSF'] + df_train['2ndFlrSF'])



df_train['Total_Bathrooms'] = (df_train['FullBath'] + (0.5 * df_train['HalfBath']) +

                               df_train['BsmtFullBath'] + (0.5 * df_train['BsmtHalfBath']))



df_train['Total_porch_sf'] = (df_train['OpenPorchSF'] + df_train['3SsnPorch'] +

                              df_train['EnclosedPorch'] + df_train['ScreenPorch'] +

                              df_train['WoodDeckSF'])







df_test['YrBltAndRemod']=df_test['YearBuilt']+df_test['YearRemodAdd']

df_test['TotalSF']=df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']



df_test['Total_sqr_footage'] = (df_test['BsmtFinSF1'] + df_test['BsmtFinSF2'] +

                                 df_test['1stFlrSF'] + df_test['2ndFlrSF'])



df_test['Total_Bathrooms'] = (df_test['FullBath'] + (0.5 * df_test['HalfBath']) +

                               df_test['BsmtFullBath'] + (0.5 * df_test['BsmtHalfBath']))



df_test['Total_porch_sf'] = (df_test['OpenPorchSF'] + df_test['3SsnPorch'] +

                              df_test['EnclosedPorch'] + df_test['ScreenPorch'] +

                              df_test['WoodDeckSF'])



df_train['haspool'] = df_train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_train['has2ndfloor'] = df_train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_train['hasgarage'] = df_train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_train['hasbsmt'] = df_train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_train['hasfireplace'] = df_train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



df_test['haspool'] = df_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_test['has2ndfloor'] = df_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_test['hasgarage'] = df_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_test['hasbsmt'] = df_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_test['hasfireplace'] = df_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error



from mlxtend.regressor import StackingCVRegressor



train_y = df_train.SalePrice

predictor_cols = ['OverallQual', 'GrLivArea', 'YrBltAndRemod','TotalSF','Total_sqr_footage','Total_Bathrooms','Total_porch_sf','LotArea', 

                 'haspool','has2ndfloor', 'hasgarage','hasbsmt' ,'hasfireplace']





train_X = df_train[predictor_cols]







kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]





lasso = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, alphas=alphas2,

                              random_state=42, cv=kfolds))





lasso_model_full_data = lasso.fit(train_X, train_y)













test_X = df_test[predictor_cols]

predicted_prices=lasso_model_full_data.predict(test_X)





print(predicted_prices)
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)
