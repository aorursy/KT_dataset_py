import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full_data = train.drop(columns = 'SalePrice').merge(test, how = 'outer').drop(columns = 'Id')
# missing values

full_data.isnull().sum()[full_data.isnull().sum() > 0]
# fill NA and 0

to_na = ['Alley', 

         'MasVnrType', 

         'BsmtQual', 

         'BsmtCond', 

         'BsmtExposure', 

         'BsmtFinType1', 

         'BsmtFinType2', 

         'FireplaceQu', 

         'GarageType', 

         'GarageFinish', 

         'GarageQual', 

         'GarageCond', 

         'PoolQC', 

         'Fence',

         'MiscFeature']

to_zero = ['MasVnrArea',

           'BsmtFinSF1',

           'BsmtFinSF2',

           'BsmtUnfSF',

           'TotalBsmtSF',

           'BsmtFullBath',

           'BsmtHalfBath'

           'GarageYrBlt']

to_zero = dict.fromkeys(to_zero, 0)

to_fill = dict.fromkeys(to_na, 'NA')

to_fill.update(to_zero)

full_data = full_data.fillna(value = to_fill)
# fill means

full_data.LotFrontage = full_data.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

full_data.MSZoning = full_data.groupby(['MSSubClass'])['MSZoning'].transform(lambda x: x.fillna(x.value_counts()[0]))



# fill modes

list_nulls = list(full_data.isnull().sum()[full_data.isnull().sum() > 0].index)

full_data[list_nulls] = full_data.groupby(['Neighborhood'])[list_nulls].transform(lambda x: x.fillna(x.value_counts().index[0]))

to_months = dict(zip([x for x in range(1,13)],

                      ['January', 'February', 

                      'March', 'April', 'May', 

                      'June', 'July', 'August', 

                      'September', 'October', 

                      'November', 'December'],))

full_data['MoSold'] = full_data['MoSold'].map(to_months)
# add new variables

full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF']+ full_data['2ndFlrSF']

full_data['HouseAge'] = full_data['YrSold'] - full_data['YearBuilt']

full_data['RemodAge'] = full_data['YrSold'] - full_data['YearRemodAdd']



# drop old and GarageYrBlt

full_data = full_data.drop(columns = ['YrSold', 'YearRemodAdd', 

                                      'YearBuilt', 'GarageYrBlt'])
# correlation heatmap

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('darkgrid')

plt.figure(figsize = (20, 18))



train_corr = full_data[full_data.dtypes[full_data.dtypes != 'object'].index]

train_corr = train_corr[:train.shape[0]]

for_pairplot = pd.concat([pd.DataFrame(train.SalePrice), train_corr], axis = 1)

train_corr = for_pairplot.corr()



sns.heatmap(train_corr, cmap = 'YlGnBu_r', annot = True).set_title('Figure 1: Correlation Heatmap for Ames Data')
# most correlated by absolute value

most_corr = train_corr['SalePrice'].map(lambda x: abs(x)).sort_values(ascending = False)[:11].index

print(most_corr)
print(full_data['GarageCars'].describe())

print(full_data['FullBath'].describe())

print(full_data['TotRmsAbvGrd'].describe())
fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (20, 27))

sns.scatterplot('TotalSF', 'SalePrice', data = for_pairplot, ax = axs[0][0])

sns.scatterplot('GrLivArea', 'SalePrice', data = for_pairplot, ax = axs[0][1])

sns.scatterplot('GarageArea', 'SalePrice', data = for_pairplot, ax = axs[1][0])

sns.scatterplot('TotalBsmtSF', 'SalePrice', data = for_pairplot, ax = axs[1][1])

sns.scatterplot('1stFlrSF', 'SalePrice', data = for_pairplot, ax = axs[2][0])

sns.scatterplot('HouseAge', 'SalePrice', data = for_pairplot, ax = axs[2][1])

plt.suptitle('Figure 2: Plots for SalePrice and Most Correlated Variables')
to_drop = [523, 1298, 581, 1190, 1061, 691, 1182, 185]

full_data = full_data.drop(to_drop)
fd_skew = full_data.skew(axis = 0).sort_values(ascending = False)

fd_skew = fd_skew.drop(['MSSubClass', 'OverallQual', 'OverallCond',

                        'HouseAge', 'RemodAge'])

print(fd_skew)
from scipy import stats

plt.figure(figsize = (12, 7))

sns.distplot(train.SalePrice, fit = stats.f).set_title('Figure 3: Distribution of SalePrice')
fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (20, 27))

sns.residplot('TotalSF', 'SalePrice', data = for_pairplot, ax = axs[0][0])

sns.residplot('GrLivArea', 'SalePrice', data = for_pairplot, ax = axs[0][1])

sns.residplot('GarageArea', 'SalePrice', data = for_pairplot, ax = axs[1][0])

sns.residplot('TotalBsmtSF', 'SalePrice', data = for_pairplot, ax = axs[1][1])

sns.residplot('1stFlrSF', 'SalePrice', data = for_pairplot, ax = axs[2][0])

sns.residplot('HouseAge', 'SalePrice', data = for_pairplot, ax = axs[2][1])

plt.suptitle('Figure 4: Residual Plots of Most Correlated Continuous Variables')
# adjust skewness with log(x+1)

skew_pos = fd_skew[fd_skew > 0.5].index

full_data[skew_pos] = full_data[skew_pos].transform(lambda x: np.log(x+1))
# create standard normal variables

to_z_score = ['HouseAge', 'RemodAge',

              'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

              'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

              'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

full_data[to_z_score] = full_data[to_z_score].transform(lambda x: (x - np.mean(x)) / np.std(x))
# Street, Alley

grav_map = {'NA' : 0, 'Grvl' : 1, 'Pave' : 2}

full_data[['Street', 'Alley']] = full_data[['Street', 'Alley']].replace(grav_map)



# LotShape 

shape_map = {'IR3' : 0, 'IR2' : 1, 'IR1' : 2, 'Reg' : 3}

full_data[['LotShape']] = full_data[['LotShape']].replace(shape_map)



# Utilities

util_map = {'ELO' : 0 , 'NoSeWa' : 1, 'NoSewr' : 2, 'AllPub' : 3}

full_data[['Utilities']] = full_data[['Utilities']].replace(util_map)



# LandSlope

slope_map = {'Sev' : 0, 'Mod' : 1, 'Gtl' : 2}

full_data[['LandSlope']] = full_data[['LandSlope']].replace(slope_map)



# HouseStyle

house_map = {'1Story' : 0, '1.5Unf' : 1, '1.5Fin' : 2, 'SFoyer' : 3, 'SLvl' : 4,

             '2Story' : 5, '2.5Unf' : 6, '2.5Fin' : 7} 

full_data[['HouseStyle']] = full_data[['HouseStyle']].replace(house_map)



# Quality

qual_map = {'NA' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5}

by_qual = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',

           'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',

           'GarageCond', 'PoolQC']

full_data[by_qual] = full_data[by_qual].replace(qual_map)



# BsmtExposure

expo_map = {'NA' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4} 

full_data['BsmtExposure'] = full_data['BsmtExposure'].replace(expo_map)



#BsmtFinType1, 2

fin_map = {'NA' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5,

           'GLQ' : 6}

full_data[['BsmtFinType1', 'BsmtFinType2']] = full_data[['BsmtFinType1', 'BsmtFinType2']].replace(fin_map)



# Electrical

elec_map = {'FuseP' : 0, 'FuseF' : 1, 'Mix' : 2, 'FuseA' : 3, 'SBkr' : 4, 'SBrkr' : 4}

full_data['Electrical'] = full_data['Electrical'].replace(elec_map)



# CentralAir

bin_map = {'N' : 0, 'No' : 0, 'Y' : 1, 'Yes' : 1}

full_data['CentralAir'] = full_data['CentralAir'].replace(bin_map)



# Functional

func_map = {'Sal' : 0, 'Sev' : 1, 'Maj2' : 2, 'Maj1' : 3, 'Mod' : 4,

            'Min2' : 5, 'Min1' : 6, 'Typ' : 7}

full_data['Functional'] = full_data['Functional'].replace(func_map)



# GarageFinish

fing_map = {'NA' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3} 

full_data['GarageFinish'] = full_data['GarageFinish'].replace(fing_map)



# PavedDrive

drive_map = {'N' : 0, 'P' : 1, 'Y' : 2} 

full_data['PavedDrive'] = full_data['PavedDrive'].replace(drive_map)



# Fence

fence_map = {'NA' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4}

full_data['Fence'] = full_data['Fence'].replace(fence_map)
# shift overallqual and cond

full_data[['OverallQual', 'OverallCond']] = full_data[['OverallQual', 'OverallCond']] - 1
to_min_max = ['Street', 'Alley', 'LotShape', 'HouseStyle', 

              'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 

              'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 

              'GarageQual', 'GarageCond', 'PoolQC', 'BsmtExposure', 

              'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'CentralAir', 

              'Functional', 'GarageFinish', 'PavedDrive', 'Fence', 

              'OverallQual', 'OverallCond', 'LandSlope']



full_data[to_min_max] = full_data[to_min_max].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# for some reason did not convert type to int earlier

full_data.CentralAir = full_data.CentralAir.astype(int)



# manual dummies

dummies_mssub = pd.get_dummies(full_data.MSSubClass).add_prefix('MSSubClass_')

dummies_mszoning = pd.get_dummies(full_data.MSZoning).add_prefix('MSZoning_')

dummies_lcontour = pd.get_dummies(full_data.LandContour).add_prefix('LandContour_')

dummies_lconfig = pd.get_dummies(full_data.LotConfig).add_prefix('LotConfig_')

dummies_month = full_data.MoSold.str.get_dummies().add_prefix('MoSold_')

dummies_bldg = full_data.BldgType.str.get_dummies().add_prefix('BldgType_')

dummies_nbr = full_data.Neighborhood.str.get_dummies().add_prefix('Neighborhood_')

dummies_cond1 = full_data.Condition1.str.get_dummies().add_prefix('Cond1_')

dummies_cond2 = full_data.Condition2.str.get_dummies().add_prefix('Cond2_')

dummies_roofs = full_data.RoofStyle.str.get_dummies().add_prefix('RoofStyle_')

dummies_roofm = full_data.RoofMatl.str.get_dummies().add_prefix('RoofMatl_')

dummies_ext1 = full_data.Exterior1st.str.get_dummies().add_prefix('Ext1_')

dummies_ext2 = full_data.Exterior2nd.str.get_dummies().add_prefix('Ext2_')

dummies_mvt = full_data.MasVnrType.str.get_dummies().add_prefix('MasVnrType_')

dummies_found = full_data.Foundation.str.get_dummies().add_prefix('Foundaton_')

dummies_heat = full_data.Heating.str.get_dummies().add_prefix('Heating_')

dummies_gart = full_data.GarageType.str.get_dummies().add_prefix('GarageType_')

dummies_misc = full_data.MiscFeature.str.get_dummies().add_prefix('Misc_')

dummies_salet = full_data.SaleType.str.get_dummies().add_prefix('SaleType_')

dummies_salec = full_data.SaleCondition.str.get_dummies().add_prefix('SaleCond_')



# out with the old

drop_vars = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 'MoSold',

             'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'RoofStyle',

             'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',

             'Heating', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']



full_data_dropped = full_data.drop(columns = drop_vars)



# in with the new

full_data_dummies = pd.concat([full_data_dropped,

                               dummies_mssub,

                               dummies_mszoning,

                               dummies_lcontour,

                               dummies_lconfig,

                               dummies_month,

                               dummies_bldg,

                               dummies_nbr,

                               dummies_cond1,

                               dummies_cond2,

                               dummies_roofs,

                               dummies_roofm,

                               dummies_ext1,

                               dummies_ext2,

                               dummies_mvt,

                               dummies_found,

                               dummies_heat,

                               dummies_gart,

                               dummies_misc,

                               dummies_salet,

                               dummies_salec], axis = 1)



print(full_data_dummies.shape)
train_clean = full_data_dummies[:(train.shape[0]-len(to_drop))]

test_clean = full_data_dummies[train_clean.shape[0]:]



print([train_clean.shape, test_clean.shape])

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_predict

from collections import namedtuple



def get_results(model, regr):

    # for error and prediction from model cross-validation

    ErrReturn = namedtuple('Output', 'Error Predictions')

    model.fit(regr, response)

    pred = cross_val_predict(model, regr, response, cv = 10)

    err = np.sqrt(mean_squared_error(pred, response))

    return ErrReturn(err, pred)



predictors = train_clean

response = np.log(train.SalePrice.drop(to_drop)) # was not transformed earlier, outliers dropped
## LASSO
from sklearn.linear_model import LassoCV

from sklearn.linear_model import Lasso

model_lasso_cv = LassoCV(alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,

                               0.1, 0.5, 1], max_iter = 50000, cv = 10)

model_lasso_cv.fit(predictors, response)

model_lasso = Lasso(alpha = model_lasso_cv.alpha_, max_iter = 50000)

results_lasso = get_results(model_lasso, predictors)



print(f"The RMSE for the LASSO model was {round(results_lasso.Error, 5)}")
# Plot important coefficients

coefs = pd.Series(model_lasso.coef_, index = predictors.columns)

print(f"LASSO picked {sum(coefs != 0)} features and eliminated the other {sum(coefs == 0)} features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

sns.barplot(x = imp_coefs.index, y = imp_coefs)

plt.title("Figure 5: Coefficients in the LASSO Model")

plt.xticks(rotation = 45,

           horizontalalignment='right')
from sklearn.linear_model import RidgeCV

from sklearn.linear_model import Ridge

model_ridge_cv = RidgeCV(alphas = [0.0001, 0.001, 0.1, 1, 5, 10, 15, 20, 25, 30])

model_ridge_cv.fit(predictors, response)

model_ridge = Ridge(alpha = model_ridge_cv.alpha_)

results_ridge = get_results(model_ridge, predictors)



print(f"The RMSE for the Ridge model was {round(results_ridge.Error, 5)}")
# Plot important coefficients

coefs = pd.Series(model_ridge.coef_, index = predictors.columns)

print(f"Ridge picked {sum(coefs != 0)} features and eliminated the other {sum(coefs == 0)} features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

sns.barplot(x = imp_coefs.index, y = imp_coefs)

plt.title("Figure 6: Coefficients in the Ridge Model")

plt.xticks(rotation = 45,

           horizontalalignment='right')
from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import ElasticNet

model_en_cv = ElasticNetCV(l1_ratio = [.1, .3, .6, .9], max_iter = 50000,

                           alphas = [0.0001, 0.001, 0.1, 1, 5, 10, 15, 20, 25, 30],

                           fit_intercept = True, cv = 10)

model_en_cv.fit(predictors, response)

model_en = ElasticNet(l1_ratio = model_en_cv.l1_ratio_,

                           alpha = model_en_cv.alpha_,

                           max_iter = 50000,

                           fit_intercept = True,

                           random_state = 1)

results_en = get_results(model_en, predictors)

print(f"The RMSE for the ElasticNet model was {round(results_en.Error, 5)}")
# Plot important coefficients

coefs = pd.Series(model_en.coef_, index = predictors.columns)

print(f"ElasticNet picked {sum(coefs != 0)} features and eliminated the other {sum(coefs == 0)} features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

sns.barplot(x = imp_coefs.index, y = imp_coefs)

plt.title("Figure 7: Coefficients in the ElasticNet Model")

plt.xticks(rotation = 45,

           horizontalalignment='right')