import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from scipy.special import boxcox1p

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Lasso, LassoCV ,ElasticNetCV,RidgeCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from mlxtend.regressor import StackingCVRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor







import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test2=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

len_train=train.shape[0]

houses=pd.concat([train,test], sort=False)

print(train.shape)

print(test.shape)
houses.select_dtypes(include='object').head()
houses.select_dtypes(include=['float','int']).head()
houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]
sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train["PoolQC"] = train["PoolQC"].fillna("None")

train["MiscFeature"] = train["MiscFeature"].fillna("None")

train["Alley"] = train["Alley"].fillna("None")

train["Fence"] = train["Fence"].fillna("None")

train["FireplaceQu"] = train["FireplaceQu"].fillna("None")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    train[col] = train[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('None')

train["MasVnrType"] = train["MasVnrType"].fillna("None")

train['MSSubClass'] = train['MSSubClass'].fillna("None")

train["Functional"] = train["Functional"].fillna("Typ")

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])

train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])

train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])

train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
test["PoolQC"] = test["PoolQC"].fillna("None")

test["MiscFeature"] = test["MiscFeature"].fillna("None")

test["Alley"] = test["Alley"].fillna("None")

test["Fence"] = test["Fence"].fillna("None")

test["FireplaceQu"] = test["FireplaceQu"].fillna("None")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    test[col] = test[col].fillna('None')



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    test[col] = test[col].fillna('None')

test["MasVnrType"] = test["MasVnrType"].fillna("None")

test["Functional"] = test["Functional"].fillna("Typ")

test['MSSubClass'] = test['MSSubClass'].fillna("None")

test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test['Utilities'] = test['Utilities'].fillna(test['MSZoning'].mode()[0])
houses.select_dtypes(include=['int','float']).isnull().sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    train[col] = train[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    train[col] = train[col].fillna(0)

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    test[col] = test[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    test[col] = test[col].fillna(0)

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)
print(train.isnull().sum().sum())

print(test.isnull().sum().sum())
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#removing outliers recomended by author

train = train[train['GrLivArea']<4000]
var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
len_train=train.shape[0]

print(train.shape)
houses=pd.concat([train,test], sort=False)

houses['_OverallQualCond'] = houses['OverallQual'] + (houses['OverallCond'] - 5) * 0.5

houses['_TotalSF'] = houses['TotalBsmtSF'] + houses['GrLivArea']

houses['_PorchArea'] = houses['OpenPorchSF'] + houses['EnclosedPorch'] + houses['3SsnPorch'] + houses['ScreenPorch']

houses['_TotalArea'] = houses['_TotalSF'] + houses['GarageArea'] + houses['_PorchArea']

houses['_Rooms'] = houses['TotRmsAbvGrd'] + houses['FullBath'] + houses['HalfBath']

houses['_BathRooms'] = houses['FullBath'] + houses['BsmtFullBath'] + (houses['HalfBath'] + houses['BsmtHalfBath']) * 0.7

houses['_GrLAreaAveByRms'] = houses['GrLivArea'] / houses['_Rooms']



houses['YrBltAndRemod']=houses['YearBuilt']+houses['YearRemodAdd']

houses['TotalSF']=houses['TotalBsmtSF'] + houses['1stFlrSF'] + houses['2ndFlrSF']



houses['Total_sqr_footage'] = (houses['BsmtFinSF1'] + houses['BsmtFinSF2'] +

                                 houses['1stFlrSF'] + houses['2ndFlrSF'])



houses['Total_Bathrooms'] = (houses['FullBath'] + (0.5 * houses['HalfBath']) +

                               houses['BsmtFullBath'] + (0.5 * houses['BsmtHalfBath']))



houses['Total_porch_sf'] = (houses['OpenPorchSF'] + houses['3SsnPorch'] +

                              houses['EnclosedPorch'] + houses['ScreenPorch'] +

                              houses['WoodDeckSF'])





houses['haspool'] = houses['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

houses['has2ndfloor'] = houses['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

houses['hasgarage'] = houses['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

houses['hasbsmt'] = houses['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

houses['hasfireplace'] = houses['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#MSSubClass=The building class

houses['MSSubClass'] = houses['MSSubClass'].apply(str)

#Year and month sold are transformed into categorical features.

houses['YrSold'] = houses['YrSold'].astype(str)

houses['MoSold'] = houses['MoSold'].astype(str)
skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skew_df=pd.DataFrame({'Skew':skew})

skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]
skewed_df.index
skewed_df = skewed_df[abs(skewed_df) > 0.5]

print("There are {} skewed numerical features to log transform".format(skewed_df.shape[0]))



for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',

       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',

       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',

       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',

       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',

       'GarageYrBlt'):

    train[col]=np.log1p(train[col])

    test[col]=np.log1p(test[col])
train=houses[:len_train]

test=houses[len_train:]
train['SalePrice']=np.log1p(train['SalePrice'])
houses=pd.concat([train,test], sort=False)

houses=pd.get_dummies(houses)
train=houses[:len_train]

test=houses[len_train:]
train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
x=train.drop('SalePrice', axis=1)

y=train['SalePrice']

test=test.drop('SalePrice', axis=1)
sc=RobustScaler()

x=sc.fit_transform(x)

test=sc.transform(test)
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



#model=Lasso(alpha =0.001, random_state=1)



modelLA=LassoCV(alphas=alphas2, random_state=42)

modelEL=ElasticNetCV(alphas=alphas2,random_state=42)

modelRI=RidgeCV(alphas=alphas2)



lgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )



xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
g_boost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, 

                                   loss='huber', random_state =5)

# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

#                              learning_rate=0.05, max_depth=3, 

#                              min_child_weight=1.7817, n_estimators=2200,

#                              reg_alpha=0.4640, reg_lambda=0.8571,

#                              subsample=0.5213, silent=1,

#                              random_state =7, nthread = -1)

# model_xgb.fit(x,y)
stack_gen = StackingCVRegressor(regressors=(modelRI,modelLA,modelEL,g_boost,lgbm,),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)

# modelEL.fit(x,y)

# modelLA.fit(x,y)

# modelRI.fit(x,y)
stack_gen_model = stack_gen.fit(np.array(x), np.array(y))


pred=stack_gen_model.predict(np.array(test))

#pred=modelEL.predict(test)

#preds = lgbm.predict(test)

preds=np.exp(pred)
output=pd.DataFrame({'Id':test2.Id, 'SalePrice':preds})

output.to_csv('StackModelV3.csv', index=False)

#output.to_csv('ElasticNetModele.csv', index=False)
#output
# Save the predictions in form of a dataframe

submission = pd.DataFrame()

submission['Id'] = test2.Id

submission['SalePrice'] = preds



top_public = pd.read_csv('../input/modele/ElasticNetModele.csv')



final_blend = (0.6*top_public.SalePrice.values + 0.4*preds)



blended_submission = pd.DataFrame()



blended_submission['Id'] = test2.Id

blended_submission['SalePrice'] = final_blend



blended_submission.to_csv('BlendModelV2.csv', index=False)
blended_submission