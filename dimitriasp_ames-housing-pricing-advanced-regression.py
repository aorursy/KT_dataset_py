import numpy as np

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import pandas_profiling



from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, LogisticRegressionCV

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFECV

from sklearn.svm import LinearSVC

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler



import sys



from yellowbrick.regressor import ResidualsPlot

from yellowbrick.features import RadViz



sys.setrecursionlimit(10000)



sns.set_style('whitegrid')



%config InlineBackend.figure_format = 'retina'

%matplotlib inline
# Load the data

house = pd.read_csv('../input/housing.csv', keep_default_na=True)

house.head()
EDA_profile_reporter = house.profile_report()
EDA_profile_reporter.to_file(output_file='EDA_profile_report.html')
house['LotFrontage'].fillna(value=0, inplace=True)

house['LotFrontage'].unique()
dict(house.isnull().sum())
house.drop(house[house['MSZoning'] == 'C (all)'].index, inplace=True, axis=0)
fixed = house[['MSZoning','LotFrontage', 'LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','YearBuilt','YearRemodAdd','Foundation','BsmtQual','BsmtExposure','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageType','GarageYrBlt','GarageCars','GarageArea','MiscFeature']]

renovatable = house.drop(fixed,axis=1)

y = house['SalePrice'].values
MSSubClass_dict = { 20:'1-STORY 1946 & NEWER ALL STYLES',

                    30:'1-STORY 1945 & OLDER',

                    40:'1-STORY W/FINISHED ATTIC ALL AGES',

                    45:'1-1/2 STORY - UNFINISHED ALL AGES',

                    50:'1-1/2 STORY FINISHED ALL AGES',

                    60:'2-STORY 1946 & NEWER',

                    70:'2-STORY 1945 & OLDER',

                    75:'2-1/2 STORY ALL AGES',

                    80:'SPLIT OR MULTI-LEVEL',

                    85:'SPLIT FOYER',

                    90:'DUPLEX - ALL STYLES AND AGES',

                   120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',

                   150:'1-1/2 STORY PUD - ALL AGES',

                   160:'2-STORY PUD - 1946 & NEWER',

                   180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',

                   190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'}
renovatable.drop(['Id','MoSold','YrSold','SaleType','SaleCondition'],axis=1)
fixed.select_dtypes(include=['object']).columns
dummies = pd.get_dummies(fixed.select_dtypes(include=['object']), dummy_na=True)
long_fixed = pd.concat([fixed,dummies], axis=1, sort=True)

long_fixed
long_fixed.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'Foundation', 'BsmtQual', 'BsmtExposure',

       'GarageType', 'MiscFeature'], axis=1, inplace=True)
dict(long_fixed.isnull().sum())
long_fixed.drop('GarageYrBlt', inplace=True, axis=1)
house.shape
long_fixed.shape
sns.distplot(house['SalePrice'] , norm_hist=True, )

plt.title('Sale Price Distribution')
house["SalePrice"]=np.log1p(house["SalePrice"])

sns.distplot(house['SalePrice'] , norm_hist=True)

plt.title('Sale Price Distribution')
train_index = house[house['YrSold'] < 2010].index

test_index = house[house['YrSold'] >= 2010].index



X_train = long_fixed.loc[train_index, :]

X_test = long_fixed.loc[test_index, :]

y_train = house.loc[train_index, 'SalePrice'].values

y_test = house.loc[test_index, 'SalePrice'].values
ss = StandardScaler()

Xs_train = ss.fit_transform(X_train)

Xs_test = ss.transform(X_test)
Xs_train.shape
lr = LinearRegression(n_jobs=-1)



lr.fit(Xs_train, y_train)

print('All of the fixed property fixtures explain', (lr.score(Xs_test, y_test))*100,'% of variance in the `Sale Price` for homes sold pre-2010.')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', alpha=0.6)

plt.scatter(y_test, lr.predict(Xs_test), alpha = .7)

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')

plt.title('Actual vs. Predicted Home Values')
lcv = LassoCV(cv=Xs_train.shape[0]-1, n_jobs=-1)



lcv.fit(Xs_train,y_train)

print('The LASSO REGRESSION model predicts', (lcv.score(Xs_test,y_test))*100,'% of the variance in `Sale Prices` on properties sold pre-2010.')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', alpha=0.6)

plt.scatter(y_test, lcv.predict(Xs_test), alpha = .7)

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')

plt.title('Actual vs. Predicted Home Values With Lassoed Features')
fixed_coef = pd.DataFrame({'Fixed Features': long_fixed.columns, 'Co-efficients': lcv.coef_, "Absolute Co-efficients":np.abs(lcv.coef_)})
fixed_coef[fixed_coef['Absolute Co-efficients']!=0].shape
sorted_fixed_coef = fixed_coef[fixed_coef['Absolute Co-efficients']!=0].sort_values('Absolute Co-efficients', ascending=False)

sorted_fixed_coef
sorted_fixed_coef.sort_values('Co-efficients').plot(x='Fixed Features',

                                                   y=['Co-efficients'],

                                                   kind='barh', 

                                                   figsize=(15,25))
a = pd.DataFrame(house.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending = True))

a.plot(figsize = (10,12),

       kind='barh',)

plt.xlabel('Sale Price (USD)')

plt.title('Median Price of Home by Neighborhood')
sorted_fixed_coef['Fixed Features'].values
lassoed_fixed_train = X_train[['GrLivArea', 'BsmtQual_Ex', 'Neighborhood_NridgHt', 'GarageCars',

       'Neighborhood_NoRidge', 'YearRemodAdd', 'YearBuilt', 'TotalBsmtSF',

       'Neighborhood_StoneBr', 'BldgType_1Fam', 'Fireplaces',

       'BsmtExposure_Gd', 'Neighborhood_Somerst', 'Condition2_PosN',

       'Neighborhood_Crawfor', 'BsmtExposure_No', 'KitchenAbvGr',

       'Neighborhood_Edwards', 'FullBath', 'BsmtFullBath', 'BsmtFinSF1',

       'LotArea', 'LotShape_IR3', 'Neighborhood_Veenker', 'MSZoning_RM',

       'Neighborhood_Mitchel', 'LandContour_Bnk', 'Condition1_Norm',

       'Condition2_PosA', 'Condition1_Feedr', 'LotShape_IR2',

       'Foundation_PConc', 'LotConfig_FR2', 'LotConfig_CulDSac',

       'GarageType_2Types', 'GarageType_CarPort', 'BldgType_Twnhs',

       'Condition1_RRAe']]
lassoed_fixed_test = X_test[['GrLivArea', 'BsmtQual_Ex', 'Neighborhood_NridgHt', 'GarageCars',

       'Neighborhood_NoRidge', 'YearRemodAdd', 'YearBuilt', 'TotalBsmtSF',

       'Neighborhood_StoneBr', 'BldgType_1Fam', 'Fireplaces',

       'BsmtExposure_Gd', 'Neighborhood_Somerst', 'Condition2_PosN',

       'Neighborhood_Crawfor', 'BsmtExposure_No', 'KitchenAbvGr',

       'Neighborhood_Edwards', 'FullBath', 'BsmtFullBath', 'BsmtFinSF1',

       'LotArea', 'LotShape_IR3', 'Neighborhood_Veenker', 'MSZoning_RM',

       'Neighborhood_Mitchel', 'LandContour_Bnk', 'Condition1_Norm',

       'Condition2_PosA', 'Condition1_Feedr', 'LotShape_IR2',

       'Foundation_PConc', 'LotConfig_FR2', 'LotConfig_CulDSac',

       'GarageType_2Types', 'GarageType_CarPort', 'BldgType_Twnhs',

       'Condition1_RRAe']]
lcv_fixed_lassoed = LassoCV(cv=lassoed_fixed_train.shape[0]-1, n_jobs=-1)



lcv_fixed_lassoed.fit(lassoed_fixed_train, y_train)

print((lcv_fixed_lassoed.score(lassoed_fixed_test,y_test))*100,'% of the variance can be explained by the remaining fixed features extracted by the Lasso Regession Model.')
plt.figure(figsize=(9, 6))

ax = sns.regplot(x=lcv_fixed_lassoed.predict(lassoed_fixed_test), y=y_test)

ax.set_xlabel('True y value')

ax.set_ylabel('Predicted y value')

ax.set_title('Variance')
residuals = y_test - lcv_fixed_lassoed.predict(lassoed_fixed_test)



plt.figure(figsize=(13, 6))

ax = sns.distplot(residuals)

ax.set_xlabel('Residuals')

ax.set_title('Distribution of Residuals');
fixed.head()
renovatable = house.drop(fixed,axis=1)

# house["SalePrice"]=np.log1p(house["SalePrice"])

y = house['SalePrice']
renovatable.columns
renovatable.drop(['Id', 'MSSubClass', 'SaleType', 'SaleCondition', 'MasVnrArea', 'SalePrice'], axis=1, inplace=True)
reno_dummies = pd.get_dummies(renovatable.select_dtypes(include=['object']), dummy_na=True, drop_first=True)
long_reno = pd.concat([renovatable,reno_dummies], axis=1, sort=True)
long_reno.drop(['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'PoolQC', 'Fence'], axis=1, inplace=True)
train_index = house[house['YrSold'] < 2010].index

test_index = house[house['YrSold'] >= 2010].index



X_reno_train = long_reno.loc[train_index, :]

X_reno_test = long_reno.loc[test_index, :]

y_reno_train = house.loc[train_index, 'SalePrice'].values

y_reno_test = house.loc[test_index, 'SalePrice'].values
ss = StandardScaler()

Xs_reno_train = ss.fit_transform(X_reno_train)

Xs_reno_test = ss.transform(X_reno_test)
lr_reno = LinearRegression(n_jobs=-1)



lr_reno.fit(X_reno_train, y_reno_train)

lr_reno.score(X_reno_test, y_reno_test)
plt.plot([min(y_reno_test), max(y_reno_test)], [min(y_reno_test), max(y_reno_test)], color='red', alpha=0.6)

plt.scatter(y_reno_test, lr_reno.predict(Xs_reno_test), alpha = .7)

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')

plt.title('Actual vs. Predicted Home Values - Renovatable Features')
lcv_reno = LassoCV(cv=Xs_reno_train.shape[0]-1, n_jobs=-1)



lcv_reno.fit(Xs_reno_train,y_reno_train)

print((lcv_reno.score(Xs_reno_test,y_reno_test))*100,'% of the variance in the `Sale Price` for properties sold pre-2010, can be explained by the Lasso Features.')
long_reno.columns
reno_coef = pd.DataFrame({'Features':long_reno.columns, 'Co-efficients': lcv_reno.coef_, "Absolute Co-efficients":np.abs(lcv_reno.coef_)})

reno_coef[reno_coef['Absolute Co-efficients']!=0].sort_values('Absolute Co-efficients', ascending=False).head(15)
house['KitchenQual'].value_counts()
house['ExterQual'].value_counts()
house['PoolQC'].value_counts()
top5_fixed_coefficients = long_fixed[['GrLivArea', 'YearRemodAdd', 'GarageCars', 'YearBuilt',

       'Neighborhood_NridgHt', 'Fireplaces', 'BsmtQual_Ex',

       'Neighborhood_Crawfor', 'BsmtFullBath', 'Neighborhood_StoneBr']]
fixed_coefficients = long_fixed[['GrLivArea', 'YearRemodAdd', 'GarageCars', 'YearBuilt',

       'Neighborhood_NridgHt', 'Fireplaces', 'BsmtQual_Ex',

       'Neighborhood_Crawfor', 'BsmtFullBath', 'Neighborhood_StoneBr',

       'Neighborhood_Edwards', 'MSZoning_RM', 'Neighborhood_Somerst',

       'FullBath', 'Condition2_PosN', 'KitchenAbvGr', 'Neighborhood_NoRidge',

       'TotalBsmtSF', 'LotArea', 'BsmtExposure_Gd', 'TotRmsAbvGrd',

       'BldgType_1Fam', 'LotShape_IR3', 'GarageType_Attchd',

       'BsmtExposure_nan', 'BldgType_Twnhs', 'Neighborhood_MeadowV',

       'BsmtExposure_No', 'GarageType_nan', 'Condition1_Feedr',

       'Condition2_PosA', 'HouseStyle_2Story', 'Foundation_PConc',

       'LandSlope_Sev', 'Neighborhood_Veenker', 'Neighborhood_Mitchel',

       'GarageType_CarPort', 'LandContour_Bnk', 'HalfBath', 'BsmtQual_Gd',

       'Neighborhood_Gilbert', 'Condition1_Artery', 'Condition1_RRAe',

       'BsmtQual_Fa', 'GarageType_2Types', 'LotConfig_CulDSac', 'BsmtFinSF1',

       'BldgType_TwnhsE', 'LotShape_IR2', 'Neighborhood_Sawyer',

       'LotConfig_FR2', 'BsmtHalfBath', 'Neighborhood_OldTown',

       'Foundation_Slab', 'BldgType_2fmCon', 'LotFrontage', 'Alley_Pave',

       'BsmtQual_nan', 'LandContour_HLS', 'Condition2_Norm',

       'Foundation_BrkTil', 'GarageArea', 'MiscFeature_TenC',

       'HouseStyle_1.5Unf', 'LowQualFinSF', 'BsmtExposure_Av',

       'Neighborhood_BrkSide', 'Neighborhood_Timber', 'Condition1_Norm',

       'Condition1_PosN', 'GarageType_BuiltIn']]
kitch_dummies = pd.get_dummies(house['KitchenQual'], prefix='KitchenQual')
kitch_dummies
combo = kitch_dummies.merge(fixed_coefficients, how='right', left_index=True, right_index=True)
combo5 = kitch_dummies.merge(top5_fixed_coefficients, how='right', left_index=True, right_index=True)
combo_full = combo.merge(house[['YrSold', 'SalePrice']], how='left', left_index=True, right_index=True)
combo_full5 = combo.merge(house[['YrSold', 'SalePrice']], how='left', left_index=True, right_index=True)
test = combo_full[combo_full['YrSold'] < 2010]

train = combo_full[combo_full['YrSold'] >= 2010]



X_combo_train = train.iloc[:, train.columns != 'SalePrice'] 

y_combo_train = train['SalePrice']

X_combo_test = test.iloc[:, train.columns != 'SalePrice']

y_combo_test = test['SalePrice']
test5 = combo_full5[combo_full5['YrSold'] < 2010]

train5 = combo_full5[combo_full5['YrSold'] >= 2010]



X_combo_train5 = train5.iloc[:, train5.columns != 'SalePrice'] 

y_combo_train5 = train5['SalePrice']

X_combo_test5 = test5.iloc[:, train5.columns != 'SalePrice']

y_combo_test5 = test5['SalePrice']
lr_combo5 = LinearRegression()

lr_combo5.fit(X_combo_train5, y_combo_train5)



print (lr_combo5.score(X_combo_test5, y_combo_test5))



a = pd.DataFrame(zip(X_combo_train5.columns, lr_combo5.coef_), columns = ['variance', 'beta'])

a['abs_beta'] = abs(a['beta'])



print (a[a['variance'] == 'KitchenQual_Ex'])

print (a[a['variance'] == 'KitchenQual_Gd'])

print (a[a['variance'] == 'KitchenQual_Ta'])

print (a[a['variance'] == 'KitchenQual_Fa'])
lr_combo = LinearRegression()

lr_combo.fit(X_combo_train, y_combo_train)



print (lr_combo.score(X_combo_test, y_combo_test))



a = pd.DataFrame(zip(X_combo_train.columns, lr_combo.coef_), columns = ['variance', 'beta'])

a['abs_beta'] = abs(a['beta'])



print (a[a['variance'] == 'KitchenQual_Ex'])

print (a[a['variance'] == 'KitchenQual_Gd'])

print (a[a['variance'] == 'KitchenQual_Ta'])

print (a[a['variance'] == 'KitchenQual_Fa'])
plt.plot([min(y_combo_test), max(y_combo_test)], [min(y_combo_test), max(y_combo_test)], c = 'r', alpha = .6)

plt.scatter(lr_combo.predict(X_combo_test), y_combo_test, alpha = .7)

plt.xlabel('actual price sold')

plt.ylabel('predicted price sold')

plt.title('Actual vs. Predicted Home Price with Kitchen Quality')
abnormal_house_df = house
abnormal_house_df.SaleCondition.value_counts(normalize=True)
abnormal_house_df['abnormal_flagging'] = abnormal_house_df['SaleCondition'].apply(lambda x: 1 if x == 'Abnorml' else 0)

target_ab = abnormal_house_df['abnormal_flagging']
abnormal_house_df.drop(columns=['SaleCondition'], axis=1, inplace=True)
abnormal_house_df.abnormal_flagging.value_counts(normalize=True)
abnormal_house_df.set_index('Id', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(abnormal_house_df, target_ab)
# OverSampling

over = RandomOverSampler(sampling_strategy=0.3)

house_ab, y_ab = over.fit_resample(X_train, y_train)

# UnderSampling

under = RandomUnderSampler(sampling_strategy=0.2)

house_ab, y_ab = under.fit_resample(X_train, y_train)
house_ab.shape
y_ab.shape
feature_ab = FeatureSelector(data=house_ab, labels=y_ab)
feature_ab.identify_zero_importance(task='classification',

                                   eval_metric='auc')
feature_ab.plot_feature_importances(plot_n=25)