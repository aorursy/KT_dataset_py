import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression ,Ridge,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

data_train = train.copy()
data_train.head()

data_train_Id = data_train.drop('Id',axis=1,inplace = True)
test_Id = test['Id']
test.drop('Id',axis=1,inplace = True)
data_train.info()
data_train.describe()
#missing data
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent*100], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#correlation matrix
corr_matrix = data_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=.8, square=True);
corr_matrix['SalePrice'].sort_values(ascending=False)
#Visualizing the most correlated features
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF',
        'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces']
sns.pairplot(data_train[cols],diag_kind='kde')
def remove_bad_features(data):
    return data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'],axis=1,inplace=True)
remove_bad_features(data_train)
fig, ax = plt.subplots() 
ax.scatter(x = data_train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
def remove_outlier(data):
    data = data[data['GrLivArea']< 5000]
    data.reset_index(drop= True,inplace=True)
    return
remove_outlier(data_train)
data_train.info()
def remove_corr_features(data):
    features = ['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageArea']
    return data.drop(features,axis=1,inplace=True)
remove_corr_features(data_train)
data_train[['LotFrontage','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical',
            'GarageType','GarageFinish','GarageQual','GarageCond']].head(20)
# Group by neighborhood and fill in missing value by the median of all the neighborhood
def imputing_missing_values(data):
        data['LotFrontage'] = data.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        data['MasVnrArea'].fillna(0,inplace=True)
        data.fillna('None',inplace=True)
        return
imputing_missing_values(data_train)
data_train.info()
cat_data = data_train.select_dtypes(object).copy()
cat_data.head()
for col in cat_data.columns:
    print(cat_data[col].unique())
def transform_categorical(data):
    global data_train
    data = pd.get_dummies(data,columns=[col for col in cat_data.columns])
    data_train = data.copy()
    return 
transform_categorical(data_train)
total_columns = data_train.copy()
total_columns =total_columns.columns
data_train.shape
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF',
        'FullBath','YearBuilt','YearRemodAdd','MasVnrArea','Fireplaces']
for col in cols:
    fig = plt.figure()

    sns.distplot(data_train[col], fit=norm);
    fig = plt.figure()
    stats.probplot(data_train[col], plot=plt)

# applying log transformation
columns = ['SalePrice','GrLivArea','TotalBsmtSF','YearBuilt','YearRemodAdd','MasVnrArea']
for col in columns :
    data_train[col+'temp'] = np.log(data_train[col].replace(0,1))

for col in columns:
    fig = plt.figure()
    sns.distplot(data_train[col+'temp'], fit=norm)
    fig = plt.figure()
    stats.probplot(data_train[col+'temp'], plot=plt)
columns_1 = ['TotalBsmtSF','YearBuilt','YearRemodAdd','MasVnrArea']
for col in columns_1 :
    data_train[col+'temp'] = np.sqrt(data_train[col].replace(0,1))

for col in columns_1:
    fig = plt.figure()
    sns.distplot(data_train[col+'temp'], fit=norm)
    fig = plt.figure()
    stats.probplot(data_train[col+'temp'], plot=plt)
# SQRT not doing well 
for col in columns:
    data_train.drop([col+'temp'],axis=1,inplace = True)
def transform_features(data):
    columns = ['SalePrice','GrLivArea']
    for col in columns:
        data[col] = np.log(data[col].replace(0,1))
    return
transform_features(data_train)
sns.distplot(data_train['SalePrice'], fit=norm)
std_scaler = StandardScaler()

def feature_scaling(data):
    global data_train
    global std_scaler
    y= data['SalePrice']
    data.drop('SalePrice',axis=1)
    data_train_array = std_scaler.fit_transform(data)
    data_ = pd.DataFrame(data_train_array , columns = data.columns , index = data.index)
    data_['SalePrice'] = y
    data_train = data_.copy()
    return
feature_scaling(data_train)
data_train.head(10)
# splitting
y = data_train['SalePrice']
data_train.drop('SalePrice', axis = 1,inplace=True)
X = data_train
X_train, X_test ,y_train, y_test = train_test_split(X.values,y.values, test_size=0.2,shuffle =True ,random_state=0)
print(X_train.shape,X_test.shape)
reg_model = LinearRegression()
reg_model.fit(X_train,y_train)
y_pred_reg = reg_model.predict(X_train)

print('the training score = ',reg_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_reg)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
def dispaly_scores(scores):
    print('scores : ', scores)
    print('mean = ', scores.mean())
    print('standard deviation = ',scores.std())

reg_score = cross_val_score(reg_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
reg_rmse_score = np.sqrt(-reg_score)
dispaly_scores(reg_rmse_score)
ridge_model = Ridge(random_state=42 , alpha=0.0005)
ridge_model.fit(X_train,y_train)
y_pred_ridge = ridge_model.predict(X_train)
print('the training score = ',ridge_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_ridge)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
ridge_score = cross_val_score(ridge_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
ridge_rmse_score = np.sqrt(-ridge_score)
dispaly_scores(ridge_rmse_score)
lasso_model = Lasso(random_state=42 , alpha=0.0005)
lasso_model.fit(X_train,y_train)
y_pred_lasso = ridge_model.predict(X_train)
print('the training score = ',lasso_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_lasso)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
lasso_score = cross_val_score(lasso_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
lasso_rmse_score = np.sqrt(-lasso_score)
dispaly_scores(lasso_rmse_score)
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train,y_train)
y_pred_tree = tree_model.predict(X_train)
print('the training score = ',tree_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_tree)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
tree_score = cross_val_score(tree_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
tree_rmse_score = np.sqrt(-tree_score)
dispaly_scores(tree_rmse_score)
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train,y_train)
y_pred_random = forest_model.predict(X_train)
print('the training score = ',forest_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_random)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
forest_score = cross_val_score(forest_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
forest_rmse_score = np.sqrt(-forest_score)
dispaly_scores(forest_rmse_score)
svr_model = SVR(kernel='rbf',C= 1)
svr_model.fit(X_train,y_train)
y_pred_svr = svr_model.predict(X_train)
print('the training score = ',svr_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_svr)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
svr_score = cross_val_score(svr_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
svr_rmse_score = np.sqrt(-svr_score)
dispaly_scores(svr_rmse_score)
gbr_model = GradientBoostingRegressor(learning_rate=1, n_estimators=1000, random_state=42)
gbr_model.fit(X_train,y_train)
y_pred_gbr = gbr_model.predict(X_train)
print('the training score = ',gbr_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_gbr)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
gbr_score = cross_val_score(gbr_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
gbr_rmse_score = np.sqrt(-gbr_score)
dispaly_scores(gbr_rmse_score)
xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213,
                             random_state =7, nthread = -1)
xgb_model.fit(X_train,y_train)
y_pred_xgb = xgb_model.predict(X_train)
print('the training score = ',xgb_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_xgb)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
xgb_score = cross_val_score(xgb_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
xgb_rmse_score = np.sqrt(-xgb_score)
dispaly_scores(xgb_rmse_score)
lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgb_model.fit(X_train,y_train)
y_pred_lgb = lgb_model.predict(X_train)
print('the training score = ',lgb_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred_lgb)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
lgb_score = cross_val_score(lgb_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
lgb_rmse_score = np.sqrt(-lgb_score)
dispaly_scores(lgb_rmse_score)
tune_pipeline_lasso = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',Lasso(random_state = 42))])

grid_search_lasso = GridSearchCV( estimator = tune_pipeline_lasso, param_grid = {'selector__k':[200,276] , 
  'model__alpha':[0.03,0.05]}, n_jobs=-1,scoring=["neg_mean_squared_error",'neg_mean_absolute_error'],
                                 refit = 'neg_mean_squared_error', cv=10, verbose=3)

grid_search_lasso.fit(X_train,y_train)
print('the best parameters : ',grid_search_lasso.best_params_)
print('the best score = ', np.sqrt(-grid_search_lasso.best_score_))
grid_search_lasso.best_estimator_.score(X_train,y_train)
tune_pipeline_tree = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',DecisionTreeRegressor(random_state = 42))])

grid_search_tree = GridSearchCV( estimator = tune_pipeline_tree, param_grid = {'selector__k':[200,276] , 
          'model__max_depth':[8,7]}, n_jobs=-1,
                scoring=["neg_mean_squared_error",'neg_mean_absolute_error'],refit = 'neg_mean_squared_error', cv=10, verbose=3)

grid_search_tree.fit(X_train,y_train)
print('the best parameters : ',grid_search_tree.best_params_)
print('the best score = ', np.sqrt(-grid_search_tree.best_score_))
grid_search_tree.best_estimator_.score(X_train,y_train)
tune_pipeline_random = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',RandomForestRegressor(random_state = 42))])

grid_search_random = GridSearchCV( estimator = tune_pipeline_random, param_grid = {'selector__k':[200,276] , 
  'model__n_estimators':np.arange(200,301,50),'model__max_depth':[15,20]}, n_jobs=-1,
                                  scoring=["neg_mean_squared_error",'neg_mean_absolute_error'],refit = 'neg_mean_squared_error', cv=10, verbose=3)

grid_search_random.fit(X_train,y_train)
print('the best parameters : ',grid_search_random.best_params_)
print('the best score = ', np.sqrt(-grid_search_random.best_score_))
grid_search_random.best_estimator_.score(X_train,y_train)
tune_pipeline_svr = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',SVR())])

grid_search_svr = GridSearchCV( estimator = tune_pipeline_svr, param_grid = {'selector__k':[200,276] , 
  'model__kernel':['linear','rbf'],'model__C':[20,100],
                                    'model__epsilon':[0.3,3]}, n_jobs=-1, scoring="neg_mean_squared_error", cv=5, verbose=3)
grid_search_svr.fit(X_train,y_train)
print('the best parameters : ',grid_search_svr.best_params_)
print('the best score = ', np.sqrt(-grid_search_svr.best_score_))
grid_search_svr.best_estimator_.score(X_train,y_train)
tune_pipeline_gbr = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',GradientBoostingRegressor(random_state=42))])

grid_search_gbr = GridSearchCV( estimator = tune_pipeline_gbr, param_grid = {'selector__k':[276] , 
  'model__loss':['huber'],'model__max_depth':[3,5],'model__learning_rate':[0.05,0.07],'model__n_estimators':[500]}, n_jobs=-1, 
                               scoring=["neg_mean_squared_error",'neg_mean_absolute_error'],refit = 'neg_mean_squared_error', cv=5, verbose=3)
grid_search_gbr.fit(X_train,y_train)
print('the best parameters : ',grid_search_gbr.best_params_)
print('the best score = ', np.sqrt(-grid_search_gbr.best_score_))
grid_search_gbr.best_estimator_.score(X_train,y_train)
tune_pipeline_ridge = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',Ridge(random_state=42))])

grid_search_ridge = GridSearchCV( estimator = tune_pipeline_ridge, param_grid = {'selector__k':[276] , 
  'model__alpha':[400,500]}, n_jobs=-1, scoring="neg_mean_squared_error", cv=5, verbose=3)
grid_search_ridge.fit(X_train,y_train)
print('the best parameters : ',grid_search_ridge.best_params_)
print('the best score = ', np.sqrt(-grid_search_ridge.best_score_))
grid_search_ridge.best_estimator_.score(X_train,y_train)
tune_pipeline_xgb = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',xgb.XGBRegressor(random_state=42))])

grid_search_xgb = GridSearchCV( estimator = tune_pipeline_xgb, param_grid = {'selector__k':[276] , 
  'model__learning_rate':[0.05,1],'model__n_estimators':[1500,1800],'model__max_depth':[3,5],'model__colsample_bytree':[0.3]},
                               n_jobs=-1, scoring="neg_mean_squared_error", cv=5, verbose=3)
grid_search_xgb.fit(X_train,y_train)
print('the best parameters : ',grid_search_xgb.best_params_)
print('the best score = ', np.sqrt(-grid_search_xgb.best_score_))
grid_search_xgb.best_estimator_.score(X_train,y_train)
tune_pipeline_lgb = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',lgb.LGBMRegressor(random_state=42,objective='regression',
                              bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11))])

grid_search_lgb = GridSearchCV( estimator = tune_pipeline_lgb, param_grid = {'selector__k':[276] , 
  'model__learning_rate':[0.005,0.01],'model__num_iterations':[10000],'model__n_estimators':[500],
                            'model__max_bin':[100],'model__num_leaves':[25,30]},
                               n_jobs=-1, scoring="neg_mean_squared_error", cv=5, verbose=3)
grid_search_lgb.fit(X_train,y_train)
print('the best parameters : ',grid_search_lgb.best_params_)
print('the best score = ', np.sqrt(-grid_search_lgb.best_score_))
grid_search_lgb.best_estimator_.score(X_train,y_train)
lasso_final_model = grid_search_lasso.best_estimator_
lasso_y_pred = lasso_final_model.predict(X_test)
lasso_final_model.score(X_test,y_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test,lasso_y_pred))
mae_lasso = mean_absolute_error(y_test,lasso_y_pred)
median_ae_lasso = median_absolute_error(y_test,lasso_y_pred)
print(rmse_lasso)
print(mae_lasso)
print(median_ae_lasso)
tree_final_model = grid_search_tree.best_estimator_
tree_y_pred = tree_final_model.predict(X_test)
tree_final_model.score(X_test,y_test)
rmse_tree = np.sqrt(mean_squared_error(y_test,tree_y_pred))
mae_tree = mean_absolute_error(y_test,tree_y_pred)
median_ae_tree = median_absolute_error(y_test,tree_y_pred)
print(rmse_tree)
print(mae_tree)
print(median_ae_tree)
random_final_model = grid_search_random.best_estimator_
random_y_pred = random_final_model.predict(X_test)
random_final_model.score(X_test,y_test)
rmse_random = np.sqrt(mean_squared_error(y_test,random_y_pred))
mae_random = mean_absolute_error(y_test,random_y_pred)
median_ae_random = median_absolute_error(y_test,random_y_pred)
print(rmse_random)
print(mae_random)
print(median_ae_random)
gbr_final_model = grid_search_gbr.best_estimator_
gbr_y_pred = gbr_final_model.predict(X_test)
gbr_final_model.score(X_test,y_test)
rmse_gbr = np.sqrt(mean_squared_error(y_test,gbr_y_pred))
mae_gbr = mean_absolute_error(y_test,gbr_y_pred)
median_ae_gbr = median_absolute_error(y_test,gbr_y_pred)
print(rmse_gbr)
print(mae_gbr)
print(median_ae_gbr)
ridge_final_model = grid_search_ridge.best_estimator_
ridge_y_pred = ridge_final_model.predict(X_test)
ridge_final_model.score(X_test,y_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test,ridge_y_pred))
mae_ridge = mean_absolute_error(y_test,ridge_y_pred)
median_ae_ridge = median_absolute_error(y_test,ridge_y_pred)
print(rmse_ridge)
print(mae_ridge)
print(median_ae_ridge)
xgb_final_model = grid_search_xgb.best_estimator_
xgb_y_pred = xgb_final_model.predict(X_test)
xgb_final_model.score(X_test,y_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test,xgb_y_pred))
mae_xgb = mean_absolute_error(y_test,xgb_y_pred)
median_ae_xgb = median_absolute_error(y_test,xgb_y_pred)
print(rmse_xgb)
print(mae_xgb)
print(median_ae_xgb)
lgb_final_model = grid_search_lgb.best_estimator_
lgb_y_pred = lgb_final_model.predict(X_test)
lgb_final_model.score(X_test,y_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test,lgb_y_pred))
mae_lgb = mean_absolute_error(y_test,lgb_y_pred)
median_ae_lgb = median_absolute_error(y_test,lgb_y_pred)
print(rmse_lgb)
print(mae_lgb)
print(median_ae_lgb)
remove_bad_features(test)
remove_corr_features(test)
def imputing_missing_values(data):
        data['LotFrontage'] = data.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        coll = data.select_dtypes('float64','int64').columns.values
        for col in coll:
            test[col].fillna(0,inplace=True)
        data.replace(np.nan,'None',inplace=True)
        return
imputing_missing_values(test)
cat_test_data = test.select_dtypes(object).copy()
def transform_categorical(data):
    global test
    data = pd.get_dummies(data,columns=[col for col in cat_test_data.columns])
    test = data.copy()
    return 
transform_categorical(test)
def transform_features(data):
    columns = ['GrLivArea']
    for col in columns:
        data[col] = np.log(data[col].replace(0,1))
    return
transform_features(test)
test.info()
# Get missing columns in the training test
missing_cols = set( total_columns.values ) - set( test.columns.values )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[total_columns.values]

extra_cols =  set( test.columns.values ) - set( total_columns.values )
for c in extra_cols:
    test.drop(c,axis=1 ,inplace = True)
test = test[total_columns.values]
    

def feature_scaling(data):
    global test
    data_test_array = std_scaler.transform(data)
    data_ = pd.DataFrame(data_test_array , columns = data.columns , index = data.index)
    test = data_.copy()
    return
feature_scaling(test)
test.drop('SalePrice',axis=1,inplace=True)
test.info()
gbr_y_test_pred = np.exp(gbr_final_model.predict(test))
xgb_y_test_pred = np.exp(gbr_final_model.predict(test))
lgb_y_test_pred = np.exp(lgb_final_model.predict(test))
prediction = gbr_y_test_pred*0.10 + xgb_y_test_pred*0.5 + lgb_y_test_pred*0.4
sub = pd.DataFrame()
sub['Id'] = test_Id
sub['SalePrice'] = prediction
sub.to_csv('submission.csv',index=False)