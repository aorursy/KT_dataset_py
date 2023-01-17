import pandas as pd

import seaborn as sns

import tensorflow as tf

import numpy as np

import math

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn import linear_model, neighbors, svm, model_selection, preprocessing, tree, feature_selection, ensemble, metrics

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

np.random.seed(1234)
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data.head()



data.head()
data.describe(include='all')
f = open('../input/house-prices-advanced-regression-techniques/data_description.txt','r')

print(f.read())

print(data.info())
missing_scores = pd.DataFrame(((data.isna().sum())/1460).sort_values(ascending=False)*100, columns=['missing values %'])

missing_scores = missing_scores[(missing_scores.T != 0).any()]

plt.figure(figsize=(10,5))

ax =sns.barplot(data = missing_scores,x = missing_scores.index, y = 'missing values %', orient='v')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()



#For 'PoolQC' NaN measn No Pool in the house...

data['PoolQC'] = data['PoolQC'].fillna('NoPool')



#For 'MiscFeature' NaN means that there is no special feature among the listed ones.

data['MiscFeature'] = data['MiscFeature'].fillna('None')



#For 'Alley' NaN means 'NoAccess', so no access to the alley

data['Alley'] = data['Alley'].fillna('NoAlley')



#For Fence means 'NoFence'

data['Fence'] = data['Fence'].fillna('NoFence')



#For Fireplace means no fireplace

data['FireplaceQu'] = data['FireplaceQu'].fillna('NoFireplace')



#For Lot frontage we have a numerical value and about 300 missing values. We could decide to substitute them with the mean value:

data['LotFrontage'] = data['LotFrontage'].fillna(0)



#For Garage Condition, garage type etc... it means no Garage

data['GarageCond'] = data['GarageCond'].fillna('NoGarage')

data['GarageType'] = data['GarageType'].fillna('NoGarage')

data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

data['GarageFinish'] = data['GarageFinish'].fillna(0)

data['GarageQual'] = data['GarageQual'].fillna('NoGarage')



#For BsmtExposure, condition etc...

data['BsmtExposure'] = data['BsmtExposure'].fillna('NoBsmt')

data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NoBsmt')

data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NoBsmt')

data['BsmtCond'] = data['BsmtCond'].fillna('NoBsmt')

data['BsmtQual'] = data['BsmtQual'].fillna('NoBsmt')



#For the other ones



data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

data['MasVnrType'] = data['MasVnrType'].fillna('None')







#For electrical data but more importantly for SalePrice we need to drop the rows with NaN values since they have no meaning.



data = data.dropna()

data = data.drop(columns=['Id'])



data.describe(include='all')
#Divide the SalePrice for 10.000 in the original dataset

data['SalePrice'] = np.log(data['SalePrice'])

y = data['SalePrice']

data = data.drop(columns=['SalePrice'])



categories = {}



#Discretize all the categorical variables

categorical_variables = data.select_dtypes('object')

for c in categorical_variables:

    data[c] = data[c].astype('category')

    categories[c] = data[c].cat.codes

    data[c] = data[c].cat.codes



#Store the info of the features in a DataFrame

info_features = pd.DataFrame({'mean': data.mean(), 'std': data.std()})

    



#zscore over all the variables

zscorer = preprocessing.StandardScaler()

zscorer.fit(data)

data = pd.DataFrame(zscorer.transform(data), columns=data.columns)
plt.figure(figsize=(10,10))

sns.clustermap(data.corr(), cmap = "Blues")
plt.figure(figsize=(20,10))

ax = sns.boxplot(data = data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()
#Perform a fitting for the lasso and plot the weights assigned to the features.

lasso = linear_model.Lasso(alpha = 0.1)

lasso_model = lasso.fit(data,y)

variables_selected = pd.DataFrame(lasso_model.coef_, index = data.columns,columns=['weight'])

variables_selected = variables_selected.sort_values(by='weight',ascending=False)

plt.figure(figsize=(15,5))

ax =sns.barplot(data = variables_selected,x = variables_selected.index, y = 'weight')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()
lasso_scores = model_selection.cross_validate(lasso,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)

print('Lasso Model scored an R squared of: mean ' + str(lasso_scores['train_r2'].mean()) + ' std: ' + str(lasso_scores['train_r2'].std()))
rand_f = ensemble.ExtraTreesRegressor()

rand_f_model = rand_f.fit(data,y)

sel_vars = pd.DataFrame(rand_f_model.feature_importances_, index = data.columns, columns=['importance'])

sel_vars = sel_vars.sort_values(by='importance', ascending=False)

plt.figure(figsize=(15,5))

ax =sns.barplot(data = sel_vars,x = sel_vars.index, y = 'importance')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()
randf_scores = model_selection.cross_validate(rand_f,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)

print('Random Forest scored an R squared of: mean ' + str(randf_scores['train_r2'].mean()) + ' std: ' + str(randf_scores['train_r2'].std()))
ridge = linear_model.Ridge(alpha=0.1)

ridge_model = ridge.fit(data,y)

variables_selected = pd.DataFrame(ridge_model.coef_, index = data.columns,columns=['weight'])

variables_selected = variables_selected.sort_values(by='weight',ascending=False)

plt.figure(figsize=(15,5))

ax =sns.barplot(data = variables_selected,x = variables_selected.index, y = 'weight')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()



ridge_scores = model_selection.cross_validate(ridge,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
lr = linear_model.LinearRegression()

lr_model = lr.fit(data,y)

lr_scores = model_selection.cross_validate(lr,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
ply = preprocessing.PolynomialFeatures(degree=2)

ply_oi = preprocessing.PolynomialFeatures(degree = 2, interaction_only = True)



ply_data = pd.DataFrame(ply.fit_transform(data), columns = ply.get_feature_names())

ply_data_oi = pd.DataFrame(ply_oi.fit_transform(data), columns = ply_oi.get_feature_names())



ply_data = ply_data[np.setdiff1d(ply_data.columns,ply_data_oi.columns)]



lr_quad = linear_model.LinearRegression()

lr_quad_model = lr_quad.fit(ply_data,y)

lr_quad_scores = model_selection.cross_validate(lr_quad,ply_data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)

selected_columns = sel_vars[sel_vars >= 0.025].dropna().index



restricted_data = data[selected_columns] #The new dataset

lr_fs = linear_model.LinearRegression()

lr_fs_model = lr_fs.fit(restricted_data,y)

lr_fs_scores = model_selection.cross_validate(lr_fs,restricted_data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
ply_fs = preprocessing.PolynomialFeatures(degree=2)

ply_fs_oi = preprocessing.PolynomialFeatures(degree=2, interaction_only = True)



ply_fs_data = pd.DataFrame(ply_fs.fit_transform(restricted_data), columns = ply_fs.get_feature_names())

ply_fs_oi_data = pd.DataFrame(ply_fs_oi.fit_transform(restricted_data),columns=ply_fs_oi.get_feature_names())

ply_fs_data = ply_fs_data[np.setdiff1d(ply_fs_data.columns,ply_fs_oi_data.columns)]

lr_fs_q = linear_model.LinearRegression()

lr_fs_q_model = lr_quad.fit(ply_fs_data,y)

lr_fs_q_scores = model_selection.cross_validate(lr_fs_q,ply_fs_data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
knn = neighbors.KNeighborsRegressor(n_neighbors = 5,weights='uniform')

knn_model = knn.fit(data,y)

knn_scores = model_selection.cross_validate(knn,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
knn_ds = neighbors.KNeighborsRegressor(n_neighbors = 5,weights='distance')

knn_ds_model = knn_ds.fit(data,y)

knn_ds_scores = model_selection.cross_validate(knn_ds,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
svr = svm.SVR(kernel='rbf')

svr_model = svr.fit(data,y)

svr_scores = model_selection.cross_validate(svr,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
gb = ensemble.GradientBoostingRegressor()

gb_model = gb.fit(data,y)

gb_scores = model_selection.cross_validate(gb,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
xgbreg = xgb.XGBRegressor()

xgb_model = xgbreg.fit(data,y)

xgb_scores = model_selection.cross_validate(xgbreg,data,y,cv=10,scoring=['r2','neg_root_mean_squared_error'], return_train_score=True)
# #Let's create the data frame for the statistics to be used for the comparisons of our models.

means_r2 = [lasso_scores['train_r2'].mean(), randf_scores['train_r2'].mean(), ridge_scores['train_r2'].mean(), lr_scores['train_r2'].mean(), lr_quad_scores['train_r2'].mean(), lr_fs_scores['train_r2'].mean(),lr_fs_q_scores['train_r2'].mean(), knn_scores['train_r2'].mean(), knn_ds_scores['train_r2'].mean(), svr_scores['train_r2'].mean(),gb_scores['train_r2'].mean(), xgb_scores['train_r2'].mean()]

stds_r2 = [lasso_scores['train_r2'].std(), randf_scores['train_r2'].std(), ridge_scores['train_r2'].std(), lr_scores['train_r2'].std(), lr_quad_scores['train_r2'].std(), lr_fs_scores['train_r2'].std(),lr_fs_q_scores['train_r2'].std(), knn_scores['train_r2'].std(), knn_ds_scores['train_r2'].mean(), svr_scores['train_r2'].std(),gb_scores['train_r2'].std(), xgb_scores['train_r2'].std()]

# mse = [metrics.mean_squared_error(y,lasso_model.predict(data)),

#        metrics.mean_squared_error(y,rand_f_model.predict(data)),

#        metrics.mean_squared_error(y,ridge_model.predict(data)),

#        metrics.mean_squared_error(y,lr_model.predict(data)),

#        metrics.mean_squared_error(y,lr_quad_model.predict(ply_data)),

#        metrics.mean_squared_error(y,lr_fs_model.predict(restricted_data)),

#        metrics.mean_squared_error(y,lr_fs_q_model.predict(ply_fs_data)),

#        metrics.mean_squared_error(y,knn_model.predict(data)),

#        metrics.mean_squared_error(y,knn_ds_model.predict(data)),

#        metrics.mean_squared_error(y,svr_model.predict(data)),

#        metrics.mean_squared_error(y,gb_model.predict(data))]

means_rmse = [lasso_scores['test_neg_root_mean_squared_error'].mean(), randf_scores['test_neg_root_mean_squared_error'].mean(), ridge_scores['test_neg_root_mean_squared_error'].mean(), lr_scores['test_neg_root_mean_squared_error'].mean(), lr_quad_scores['test_neg_root_mean_squared_error'].mean(), lr_fs_scores['test_neg_root_mean_squared_error'].mean(),lr_fs_q_scores['test_neg_root_mean_squared_error'].mean(), knn_scores['test_neg_root_mean_squared_error'].mean(), knn_ds_scores['test_neg_root_mean_squared_error'].mean(), svr_scores['test_neg_root_mean_squared_error'].mean(),gb_scores['test_neg_root_mean_squared_error'].mean(), xgb_scores['test_neg_root_mean_squared_error'].mean()]

stds_rmse = [lasso_scores['test_neg_root_mean_squared_error'].std(), randf_scores['test_neg_root_mean_squared_error'].std(), ridge_scores['test_neg_root_mean_squared_error'].std(), lr_scores['test_neg_root_mean_squared_error'].std(), lr_quad_scores['test_neg_root_mean_squared_error'].std(), lr_fs_scores['test_neg_root_mean_squared_error'].std(),lr_fs_q_scores['test_neg_root_mean_squared_error'].std(), knn_scores['test_neg_root_mean_squared_error'].std(), knn_ds_scores['test_neg_root_mean_squared_error'].mean(), svr_scores['test_neg_root_mean_squared_error'].std(),gb_scores['test_neg_root_mean_squared_error'].std(), xgb_scores['test_neg_root_mean_squared_error'].std()]





model_scores = pd.DataFrame({'model': ['lasso','random_forest','ridge','lr','lr_quad','lr_fs','lr_fs_q','knn','knn_ds','svr','gb','xgb'], 'mean_r2': means_r2, 'std_r2': stds_r2,

                            'means_RMSE': means_rmse,'std_RMSE':stds_rmse})

model_scores
plt.ylim([-0.33,0])

plt.xticks(rotation=60)

plt.axhline(y=max(model_scores['means_RMSE']))

sns.barplot(data = model_scores, x='model', y='means_RMSE')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_data.head()
test_data.columns[np.where(test_data.isna().any() == True)]
#For 'PoolQC' NaN measn No Pool in the house...

test_data['PoolQC'] = test_data['PoolQC'].fillna('NoPool')



#For 'MiscFeature' NaN means that there is no special feature among the listed ones.

test_data['MiscFeature'] = test_data['MiscFeature'].fillna('None')



#For 'Alley' NaN means 'NoAccess', so no access to the alley

test_data['Alley'] = test_data['Alley'].fillna('NoAlley')



#For Fence means 'NoFence'

test_data['Fence'] = test_data['Fence'].fillna('NoFence')



#For Fireplace means no fireplace

test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('NoFireplace')



#For Lot frontage we have a numerical value and about 300 missing values. We could decide to substitute them with the mean value:

test_data['LotFrontage'] = test_data['LotFrontage'].fillna(0)



#For Garage Condition, garage type etc... it means no Garage

test_data['GarageCond'] = test_data['GarageCond'].fillna('NoGarage')

test_data['GarageType'] = test_data['GarageType'].fillna('NoGarage')

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(0)

test_data['GarageFinish'] = test_data['GarageFinish'].fillna(0)

test_data['GarageQual'] = test_data['GarageQual'].fillna('NoGarage')



#For BsmtExposure, condition etc...

test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna('NoBsmt')

test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna('NoBsmt')

test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna('NoBsmt')

test_data['BsmtCond'] = test_data['BsmtCond'].fillna('NoBsmt')

test_data['BsmtQual'] = test_data['BsmtQual'].fillna('NoBsmt')



#For the other ones



test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)

test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')



#For electrical data but more importantly for SalePrice we need to drop the rows with NaN values since they have no meaning.

test_data.drop(columns=['Id'],inplace=True)



test_data.describe(include='all')
test_data.columns[np.where(test_data.isna().any() == True)]
for c in categorical_variables:

    test_data[c] = test_data[c].astype('category')

    test_data[c] = categories[c]

test_data.head()

test_data.astype('double')
test_data = test_data.fillna(-1) #In this way we consider as negative the missing values for the predictions
#Now standardize the data from the original distribution

test_data = pd.DataFrame(zscorer.transform(test_data), columns = test_data.columns)
test_data.head()
test_data.isna().sum()
predictions_gb = gb_model.predict(test_data)

predictions_rf = rand_f_model.predict(test_data)

predictions_xgb = xgb_model.predict(test_data)



restricted_data_test = test_data[selected_columns] #The new dataset 

ply_test = preprocessing.PolynomialFeatures(degree=2)

ply_test_oi = preprocessing.PolynomialFeatures(degree=2, interaction_only = True)



ply_test_data = pd.DataFrame(ply_test.fit_transform(restricted_data_test), columns = ply_test.get_feature_names())

ply_test_oi_data = pd.DataFrame(ply_test_oi.fit_transform(restricted_data_test), columns = ply_test_oi.get_feature_names())



ply_test_data = ply_test_data[np.setdiff1d(ply_test_data.columns,ply_test_oi_data.columns)]

predictions_lrqfs = lr_fs_q_model.predict(ply_test_data)
sub1 = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub2 = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub3 = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub4 = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub1['SalePrice'] = np.exp(predictions_gb).astype(np.int64)

sub2['SalePrice'] = np.exp(predictions_rf).astype(np.int64)

sub3['SalePrice'] = np.exp(predictions_lrqfs).astype(np.int64)

sub4['SalePrice'] = np.exp(predictions_xgb).astype(np.int64)
sub1.set_index('Id',inplace=True)

sub2.set_index('Id',inplace=True)

sub3.set_index('Id',inplace=True)

sub4.set_index('Id',inplace=True)
sub1.to_csv('submission_gb.csv') ## Score: 0.16

sub2.to_csv('submission_rf.csv') ## Score: 0.20

sub3.to_csv('submission_lrqfs.csv') ## Score: 0.22

sub4.to_csv('submission_xgb.csv') ## Score: ?