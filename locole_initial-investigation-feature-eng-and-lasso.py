# General libraries

import numpy as np

import pandas as pd



# For plotting

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

%matplotlib inline



# For data prep

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# For learning

from sklearn.linear_model import LinearRegression, LassoCV



# For Testing

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



print('Imported training data has', df_train.shape[0], 'entries')

print('Imported testing data has', df_test.shape[0], 'entries')
df_train.info()
sns.set()

x = df_train['SalePrice']

ax = sns.distplot(x, fit=norm, kde=False, color='xkcd:green blue')
sns.set()

x = np.log(df_train['SalePrice'].copy())

ax = sns.distplot(x, fit=norm, kde=False, color='xkcd:green blue', axlabel = 'Log(SalePrice)')
df_train['SalePrice'] = np.log(df_train['SalePrice'])

target_train = df_train['SalePrice']



# Rename column for clarity

df_train.rename(columns={'SalePrice': 'LogSalePrice'}, inplace=True)
cat_cols = df_train.columns[df_train.dtypes == 'object']

num_cols = df_train.columns[(df_train.dtypes == 'float64') | (df_train.dtypes == 'int64')]



# Convert all numerical values into float type

for col in num_cols:

    df_train[col] = df_train[col].astype(float)
df_basic = df_train.copy()

df_basic = pd.get_dummies(df_basic)



for col in df_basic.columns:

    df_basic[col] = df_basic[col].fillna(df_basic[col].mean())

    

# check nulls filled

print('Total null values:', (df_basic.isnull()*1).sum().sum())
yb = df_basic['LogSalePrice']



cols = df_basic.columns

cols = cols.drop('LogSalePrice')

Xb = df_basic[cols]



Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.25, random_state=123)



# Apply standard scaler to explanatory variables



stdsc_b = StandardScaler()

Xb_train_std = stdsc_b.fit_transform(Xb_train)

Xb_test_std = stdsc_b.transform(Xb_test)



# Check standard scaling is working as expected



print('Maximum absolute mean of transformed data is',abs(Xb_train_std.mean(axis=0)).max())

print('Maximum standard deviation of transformed data is', Xb_train_std.std(axis=0).max())

print('Minimum standard deviation of transformed data is', Xb_train_std.std(axis=0).min())
lin_model = LinearRegression()

lin_model.fit(Xb_train_std, yb_train)

yb_train_pred = lin_model.predict(Xb_train_std)

yb_test_pred = lin_model.predict(Xb_test_std)
print('R^2 score:', r2_score(yb_train,yb_train_pred))

print('MSE:', mean_squared_error(yb_train, yb_train_pred))

print('rMSE:', np.sqrt(mean_squared_error(yb_train, yb_train_pred)))

print('Mean Absolute Error:', mean_absolute_error(yb_train,yb_train_pred))

print('Max Over-estimation %age:', ((yb_train_pred - yb_train)/yb_train).max())

print('Max Under-estimation %age:', ((yb_train - yb_train_pred)/yb_train).max())
print('R^2 score:', r2_score(yb_test,yb_test_pred))

print('MSE:', mean_squared_error(yb_test, yb_test_pred))

print('rMSE:', np.sqrt(mean_squared_error(yb_test, yb_test_pred)))

print('Mean Absolute Error:', mean_absolute_error(yb_test,yb_test_pred))

print('Max Over-estimation %age:', ((yb_test_pred - yb_test)/yb_test).max())

print('Max Under-estimation %age:', ((yb_test - yb_test_pred)/yb_test).max())
basic_lasso = LassoCV()

basic_lasso.fit(Xb_train_std, yb_train)

yb_train_pred = basic_lasso.predict(Xb_train_std)

yb_test_pred = basic_lasso.predict(Xb_test_std)
print('R^2 score:', r2_score(yb_train,yb_train_pred))

print('MSE:', mean_squared_error(yb_train, yb_train_pred))

print('rMSE:', np.sqrt(mean_squared_error(yb_train, yb_train_pred)))

print('Mean Absolute Error:', mean_absolute_error(yb_train,yb_train_pred))

print('Max Over-estimation %age:', ((yb_train_pred - yb_train)/yb_train).max())

print('Max Under-estimation %age:', ((yb_train - yb_train_pred)/yb_train).max())
print('R^2 score:', r2_score(yb_test,yb_test_pred))

print('MSE:', mean_squared_error(yb_test, yb_test_pred))

print('rMSE:', np.sqrt(mean_squared_error(yb_test, yb_test_pred)))

print('Mean Absolute Error:', mean_absolute_error(yb_test,yb_test_pred))

print('Max Over-estimation %age:', ((yb_test_pred - yb_test)/yb_test).max())

print('Max Under-estimation %age:', ((yb_test - yb_test_pred)/yb_test).max())
# Create benchmark rMSE



IS_rMSE = np.sqrt(mean_squared_error(yb_train, yb_train_pred))

OS_rMSE = np.sqrt(mean_squared_error(yb_test, yb_test_pred))
ax = sns.violinplot(x="PoolQC", y="LogSalePrice", data=df_train, cut=0, color='xkcd:green blue')
df_train['PoolQC'].value_counts()
del df_train['PoolQC']

cat_cols = cat_cols.drop('PoolQC')
cat_vals = df_train[cat_cols]

num_mode_vals = ((cat_vals == cat_vals.mode().values)*1).sum()

num_mode_vals[num_mode_vals > 0.98 * df_train.shape[0]] / df_train.shape[0]
f, axarr = plt.subplots(2,2, figsize = (15,15))

sns.violinplot(x='Street', y='LogSalePrice', data=df_train, cut=0, ax=axarr[0,0], color='xkcd:green blue')

sns.violinplot(x='Utilities', y='LogSalePrice', data=df_train, cut=0, ax=axarr[0,1], color='xkcd:green blue')

sns.violinplot(x='Condition2', y='LogSalePrice', data=df_train, cut=0, ax=axarr[1,0], color='xkcd:green blue')

sns.violinplot(x='RoofMatl', y='LogSalePrice', data=df_train, cut=0, ax=axarr[1,1], color='xkcd:green blue')

plt.show()
del df_train['Street']

del df_train['Utilities']

del df_train['Condition2']

del df_train['RoofMatl']



cat_cols = cat_cols.drop(['Street', 'Utilities', 'Condition2','RoofMatl'])
df_train[df_train.columns[df_train.isnull().any()].tolist()].isnull().sum()
df_train.loc[df_train['LotFrontage'].isnull(),'LotFrontage'] = df_train['LotFrontage'].mean()

df_train.loc[df_train['Alley'].isnull(),'Alley'] = 'None'

df_train.loc[df_train['MasVnrType'].isnull(),'MasVnrType'] = 'None'

df_train.loc[df_train['MasVnrArea'].isnull(),'MasVnrArea'] = 0

df_train.loc[df_train['BsmtQual'].isnull(),'BsmtQual'] = 'NoBsmt'

df_train.loc[df_train['BsmtCond'].isnull(),'BsmtCond'] = 'NoBsmt'

df_train.loc[df_train['BsmtExposure'].isnull(),'BsmtExposure'] = 'NoBsmt'

df_train.loc[df_train['BsmtFinType1'].isnull(),'BsmtFinType1'] = 'NoBsmt'

df_train.loc[df_train['BsmtFinType2'].isnull(),'BsmtFinType2'] = 'NoBsmt'

df_train.loc[df_train['Electrical'].isnull(),'Electrical'] = df_train['Electrical'].mode()[0]

df_train.loc[df_train['FireplaceQu'].isnull(),'FireplaceQu'] = 'NoFire'

df_train.loc[df_train['GarageType'].isnull(),'GarageType'] = 'NoGarage'

df_train.loc[df_train['GarageYrBlt'].isnull(),'GarageYrBlt'] = 0

df_train.loc[df_train['GarageFinish'].isnull(),'GarageFinish'] = 'NoGarage'

df_train.loc[df_train['GarageQual'].isnull(),'GarageQual'] = 'NoGarage'

df_train.loc[df_train['GarageCond'].isnull(),'GarageCond'] = 'NoGarage'

df_train.loc[df_train['Fence'].isnull(),'Fence'] = 'NoFence'

df_train.loc[df_train['MiscFeature'].isnull(),'MiscFeature'] = 'NoFeat'



# Check that all null values taken care of

df_train[df_train.columns[df_train.isnull().any()].tolist()].isnull().sum()
# Define mappings for ordinal variables



LotShape_Map = {

    'Reg': 4,

    'IR1': 3,

    'IR2': 2,

    'IR3': 1

}



LandContour_Map = {

    'Lvl': 4,

    'Bnk': 3,

    'HLS': 2,

    'Low': 1

}



LotConfig_Map = {

    'Inside': 5,

    'Corner': 4,

    'CulDSac': 3,

    'FR2': 2,

    'FR3': 1

}



LandSlope_Map = {

    'Gtl': 1,

    'Mod': 2,

    'Sev': 3

}



BldgType_Map = {

    '1Fam': 5,

    '2fmCon': 4,

    'Duplex': 3,

    'TwnhsE': 2,

    'Twnhs': 1

}



HouseStyle_Map = {

    '1Story': 1,

    '1.5Fin': 3,

    '1.5Unf': 2,

    '2Story': 4,

    '2.5Fin': 6,

    '2.5Unf': 5,

    'SFoyer': 7,

    'SLvl': 8

    

}



BsmtExposure_Map = {

    'Gd': 4,

    'Av': 3,

    'Mn': 2,

    'No': 1,

    'NoBsmt': 0

}



BsmtFinType_Map = {

    'GLQ': 6,

    'ALQ': 5,

    'BLQ': 4,

    'Rec': 3,

    'LwQ': 2,

    'Unf': 1,

    'NoBsmt': 0

}



CentralAir_Map = {

    'N': 0,

    'Y': 1

}



Electrical_Map = {

    'SBrkr': 5,

    'FuseA': 4,

    'FuseF': 3,

    'FuseP': 2,

    'Mix': 1

}



Functional_Map = {

    'Typ': 8,

    'Min1': 7,

    'Min2': 6,

    'Mod': 5,

    'Maj1': 4,

    'Maj2': 3,

    'Sev': 2,

    'Sal': 1

}



GarageFinish_Map = {

    'Fin': 3,

    'RFn': 2,

    'Unf': 1,

    'NoGarage': 0

}



PavedDrive_Map = {

    'Y': 3,

    'P': 2,

    'N': 1

}



Fence_Map = {

    'GdPrv': 4,

    'MnPrv': 3,

    'GdWo': 2,

    'MnWw': 1,

    'NoFence': 0

}



Qual_Cond_Map = {

    'Po': 1,

    'Fa': 2,

    'TA': 3,

    'Gd': 4,

    'Ex': 5

}



Bsmt_Qual_Cond_Map = {

    'NoBsmt': 0,

    'Po': 1,

    'Fa': 2,

    'TA': 3,

    'Gd': 4,

    'Ex': 5

}



Fire_Qual_Cond_Map = {

    'NoFire': 0,

    'Po': 1,

    'Fa': 2,

    'TA': 3,

    'Gd': 4,

    'Ex': 5

}



Garage_Qual_Cond_Map = {

    'NoGarage': 0,

    'Po': 1,

    'Fa': 2,

    'TA': 3,

    'Gd': 4,

    'Ex': 5

}





# Apply Mappings to convert categorical ordinal variables. 

# NA values are dealt with in the natural way based on the data description file and the mode taken otherwise



df_train['LotShape'] = df_train['LotShape'].map(LotShape_Map)

df_train['LandContour'] = df_train['LandContour'].map(LandContour_Map)

df_train['LotConfig'] = df_train['LotConfig'].map(LotConfig_Map)

df_train['LandSlope'] = df_train['LandSlope'].map(LandSlope_Map)

df_train['BldgType'] = df_train['BldgType'].map(BldgType_Map)

df_train['HouseStyle'] = df_train['HouseStyle'].map(HouseStyle_Map)

df_train['ExterQual'] = df_train['ExterQual'].map(Qual_Cond_Map)

df_train['ExterCond'] = df_train['ExterCond'].map(Qual_Cond_Map)

df_train['BsmtQual'] = df_train['BsmtQual'].map(Bsmt_Qual_Cond_Map)

df_train['BsmtCond'] = df_train['BsmtCond'].map(Bsmt_Qual_Cond_Map)

df_train['BsmtExposure'] = df_train['BsmtExposure'].map(BsmtExposure_Map)

df_train['BsmtFinType1'] = df_train['BsmtFinType1'].map(BsmtFinType_Map)

df_train['BsmtFinType2'] = df_train['BsmtFinType2'].map(BsmtFinType_Map)

df_train['HeatingQC'] = df_train['HeatingQC'].map(Qual_Cond_Map)

df_train['CentralAir'] = df_train['CentralAir'].map(CentralAir_Map)

df_train['Electrical'] = df_train['Electrical'].map(Electrical_Map)

df_train['KitchenQual'] = df_train['KitchenQual'].map(Qual_Cond_Map)

df_train['Functional'] = df_train['Functional'].map(Functional_Map)

df_train['FireplaceQu'] = df_train['FireplaceQu'].map(Fire_Qual_Cond_Map)

df_train['GarageFinish'] = df_train['GarageFinish'].map(GarageFinish_Map)

df_train['GarageQual'] = df_train['GarageQual'].map(Garage_Qual_Cond_Map)

df_train['GarageCond'] = df_train['GarageCond'].map(Garage_Qual_Cond_Map)

df_train['PavedDrive'] = df_train['PavedDrive'].map(PavedDrive_Map)

df_train['Fence'] = df_train['Fence'].map(Fence_Map)



# Check mappings haven't induced more null entries

df_train[df_train.columns[df_train.isnull().any()].tolist()].isnull().sum()
MSSubClass_Map = {

    20: '1-STORY 1946 & NEWER ALL STYLES',

    30: '1-STORY 1945 & OLDER',

    40: '1-STORY W/FINISHED ATTIC ALL AGES',

    45: '1-1/2 STORY - UNFINISHED ALL AGES',

    50: '1-1/2 STORY FINISHED ALL AGES',

    60: '2-STORY 1946 & NEWER',

    70: '2-STORY 1945 & OLDER',

    75: '2-1/2 STORY ALL AGES',

    80: 'SPLIT OR MULTI-LEVEL',

    85: 'SPLIT FOYER',

    90: 'DUPLEX - ALL STYLES AND AGES',

   120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',

   150: '1-1/2 STORY PUD - ALL AGES',

   160: '2-STORY PUD - 1946 & NEWER',

   180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',

   190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'



}



df_train['MSSubClass'] = df_train['MSSubClass'].map(MSSubClass_Map)

df_train['MSSubClass'] = df_train['MSSubClass'].map(MSSubClass_Map)
df_train = pd.get_dummies(df_train)
# Drop MSSubClass from num_cols as this was actually categorical and has been dealt with

num_cols = num_cols.drop('MSSubClass')



# Find columns with kurtosis greater than 2 or less the -2. Ignore 0 values which tend to relate to 

# the absense of a feature (e.g. 0 basement area relates to no basement)

kurt_ind = num_cols[abs(df_train[num_cols][df_train[num_cols] != 0].kurtosis()) > 2]



# Show kurtosis for numerical variables

df_train[num_cols][df_train[num_cols] != 0].kurtosis()[kurt_ind].sort_values(ascending=False)
print('Largest 20 LotArea values:')

print(df_train['LotArea'].sort_values(ascending=False).head(20))

print('Mean LotArea value', df_train['LotArea'].mean())

print('Median LotArea value', df_train['LotArea'].median())
area_cols = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

area_vals = df_train[area_cols].copy()
area_vals[area_vals !=0].kurtosis().sort_values(ascending=False)
# Apply log(1+x) transform (to avoid errors with 0 values)

area_vals = np.log1p(area_vals)

area_vals[area_vals !=0].kurtosis().sort_values(ascending=False)
df_train[area_cols] = np.log1p(df_train[area_cols])

df_train[area_cols] = np.log1p(df_train[area_cols])
# Re-label columns so it is clear transformation has taken place



for col in area_cols:

    df_train.rename(columns={col: 'Log'+col}, inplace=True)
# Find top 20 strongest correlation relations (positive or negative) and show on a heatmap



full_corr_mat = df_train.corr().abs()

corr_cols = full_corr_mat.nlargest(20, 'LogSalePrice')['LogSalePrice'].index

corr_mat = np.corrcoef(df_train[corr_cols].values.T)

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat, yticklabels=corr_cols.values, xticklabels=corr_cols.values, vmin=0, vmax=1, annot=True, cmap="BuGn")

plt.show()
sig_cols = ['LogSalePrice', 'OverallQual', 'LogGrLivArea', 'GarageCars', 'ExterQual', 'KitchenQual', 'BsmtQual']

sns.set()

sns.pairplot(df_train[sig_cols])

plt.show()
del df_train['TotRmsAbvGrd']

del df_train['Fireplaces']
print('Have',df_train.columns.size - 1, 'explanatory features')
y = df_train['LogSalePrice']



cols = df_train.columns

cols = cols.drop('LogSalePrice')

X = df_train[cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
# Apply standard scaler to explanatory variables



stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)
# Check standard scaling is working as expected



print('Maximum absolute mean of transformed data is',abs(X_train_std.mean(axis=0)).max())

print('Maximum standard deviation of transformed data is', X_train_std.std(axis=0).max())

print('Minimum standard deviation of transformed data is', X_train_std.std(axis=0).min())
model_lasso = LassoCV()

model_lasso.fit(X_train_std, y_train)

print('Alpha parameter selected:', model_lasso.alpha_)
y_train_pred = model_lasso.predict(X_train_std)

y_test_pred = model_lasso.predict(X_test_std)
print('R^2 score:', r2_score(y_train,y_train_pred))

print('MSE:', mean_squared_error(y_train, y_train_pred))

print('rMSE:', np.sqrt(mean_squared_error(y_train, y_train_pred)))

print('Mean Absolute Error:', mean_absolute_error(y_train,y_train_pred))

print('Max Over-estimation %age:', ((y_train_pred - y_train)/y_train).max())

print('Max Under-estimation %age:', ((y_train - y_train_pred)/y_train).max())
# Compare rMSE to benchmark



print('In sample rMSE has changed by',-(IS_rMSE - (np.sqrt(mean_squared_error(y_train, y_train_pred))))/IS_rMSE*100, '%')
print('R^2 score:', r2_score(y_test,y_test_pred))

print('MSE:', mean_squared_error(y_test, y_test_pred))

print('rMSE:', np.sqrt(mean_squared_error(y_test, y_test_pred)))

print('Mean Absolute Error:', mean_absolute_error(y_test,y_test_pred))

print('Max Over-estimation %age:', ((y_test_pred - y_test)/y_test).max())

print('Max Under-estimation %age:', ((y_test - y_test_pred)/y_test).max())
print('In sample rMSE has changed by',-(OS_rMSE - (np.sqrt(mean_squared_error(y_test, y_test_pred))))/OS_rMSE*100, '%')
f, axarr = plt.subplots(1, figsize = (10,10))

plt.scatter(y_train_pred,

           y_train_pred - y_train,

           c='xkcd:green blue',

           marker='o',

           label='Training Data')

plt.scatter(y_test_pred,

           y_test_pred - y_test,

           c='xkcd:pumpkin',

           marker='+',

           label='Test Data')

plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc='upper left')

plt.hlines(y=0, xmin=10, xmax=14, color='black')

plt.xlim([10,14])

plt.show()
stdsc2 = StandardScaler()

X_std = stdsc2.fit_transform(X)

model_lasso2 = LassoCV()

model_lasso2.fit(X_std, y)

y_pred = model_lasso2.predict(X_std)
print('R^2 score:', r2_score(y,y_pred))

print('MSE:', mean_squared_error(y, y_pred))

print('rMSE:', np.sqrt(mean_squared_error(y, y_pred)))

print('Mean Absolute Error:', mean_absolute_error(y,y_pred))

print('Max Over-estimation %age:', ((y_pred - y)/y).max())

print('Max Under-estimation %age:', ((y - y_pred)/y).max())
model_lasso_cv = make_pipeline(StandardScaler(), LassoCV())

k = 50

rMSE_cv = np.sqrt(-cross_val_score(model_lasso_cv, X, y, cv=k, scoring="neg_mean_squared_error"))



print('Mean rMSE:', rMSE_cv.mean())

print('Standard Deviation rMSE:', rMSE_cv.std())
coef = pd.Series(model_lasso2.coef_, index = X_train.columns)

pos_imp_coef = coef.sort_values(ascending=False).head(10).index

neg_imp_coef = coef.sort_values(ascending=True).head(10).index





f, axarr = plt.subplots(2, figsize = (10,10), sharex=True)

sns.barplot(x=coef[pos_imp_coef], y=pos_imp_coef, color='xkcd:green blue', ax=axarr[0])

sns.barplot(x=coef[neg_imp_coef], y=neg_imp_coef, color='xkcd:green blue', ax=axarr[1])

axarr[0].set_title('10 Most Significant Positive Features')

axarr[1].set_title('10 Most Significant Negative Features')

plt.show()
print('Lasso has removed',coef[coef != 0].size, 'features. Leaving', coef.size-coef[coef != 0].size, 'features')