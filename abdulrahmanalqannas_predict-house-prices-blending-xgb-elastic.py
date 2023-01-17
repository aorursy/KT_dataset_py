import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
#sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
# display all the dateframe
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col='Id')
# combine train and test data 
df_all = pd.concat([train_data.drop('SalePrice', axis=1), test_data], sort=True)  #df without the target
df_all.head()
df_all.shape
df_all.info()
numeric_data = df_all.select_dtypes(include=['int64','float64']) # by default pandas read data in 64 bit
categorical_data = df_all.select_dtypes(include=['object'])
print('Numeric Features:',len(list(numeric_data.columns)))
print('Categorical Features:',len(list(categorical_data.columns)))
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
sns.heatmap(train_data.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Trian data')

sns.heatmap(test_data.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');
missing = df_all.isnull().sum().sort_values(ascending=False)
percentage = (df_all.isnull().sum() / df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%']) 
missing_data = pd.concat([missing_data, df_all.dtypes], axis=1, join='inner').rename(columns={0:'type'})
missing_data = missing_data[missing_data != 0].dropna()
missing_data
missing_data[missing_data['type'] != 'object']
missing_data[missing_data['type'] == 'object']
# adding the target to our df
df_all = pd.concat([df_all, train_data['SalePrice']], axis=1)
normal_sp = df_all['SalePrice'].dropna().map(lambda i: np.log(i) if i > 0 else 0)
print(df_all['SalePrice'].skew())
print(normal_sp.skew())

fig, ax = plt.subplots(ncols=2, figsize=(12,6))
df_all.hist('SalePrice', ax=ax[0])
normal_sp.hist(ax=ax[1])
plt.show();
df_all['SalePrice'].describe()
corr = df_all.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=1, square=True, cmap=sns.diverging_palette(180, 10, as_cmap = True));
# correlation with the target
corr_matrix = df_all.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 8))
corrmat = abs(df_all.corr())
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cols = corrmat[corrmat['SalePrice'] >= 0.5].index    # use this line or the line above
cm = abs(np.corrcoef(df_all[cols].dropna().values.T))

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
            yticklabels=cols.values, xticklabels=cols.values, ax=ax, cmap="YlGnBu")
plt.title('The highest correlated numeric features with Sale Price')
plt.show();
fig, axes = plt.subplots(ncols=4, nrows=4, 
                         figsize=(5 * 5, 5 * 5), sharey=True)
axes = np.ravel(axes)
cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',
        'BsmtCond','GarageQual','GarageCond', 'MSSubClass','MSZoning',
        'Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']

for i, c in zip(np.arange(len(axes)), cols):
    ax = sns.boxplot(x=c, y='SalePrice', data=df_all, ax=axes[i], palette="Set2")
    ax.set_title(c)
    ax.set_xlabel("")
object_features = df_all.loc[:, df_all.dtypes == np.object] 
for col in object_features.columns:
    print(df_all[col].value_counts(), '\n\n\n\n')
df_all['GarageYrBlt'].fillna(df_all['YearBuilt'], inplace = True)
# for feat in ['YearBuilt','YearRemodAdd', 'GarageYrBlt']:
#     df_all[feat] = df_all['YrSold'] - df_all[feat]
# new features to indicate the null values
# for feat in missing_data[missing_data['type'] != 'object'].index:   # index here contains columns names
#     df_all[feat+'_NaN'] = np.where(df_all[feat].isnull(), 1, 0)
# imputer = KNNImputer(n_neighbors=60)
# df_all.loc[:, df_all.dtypes != np.object] = imputer.fit_transform(numeric_data)
# imp = SimpleImputer(missing_values=np.nan, strategy='median')
# df_all.loc[:, df_all.dtypes != np.object] = imp.fit_transform(numeric_data)
df_all[numeric_data.columns] = df_all[numeric_data.columns].interpolate(method='linear')
df_all[numeric_data.columns].isnull().sum().sum()
edit_values = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType',
              'GarageQual','PoolQC','Fence','MiscFeature','MasVnrType', 'GarageCond', 'GarageFinish']

df_all[edit_values] = df_all[edit_values].fillna('NA')
catg_nulls = df_all[categorical_data.columns].isnull().sum()[df_all.isnull().sum() > 0]
print(f'Total Missing values: {catg_nulls.sum()} \n\n{catg_nulls}')
# _ = [df_all[col].fillna(df_all[col].mode()[0], inplace=True) for col in catg_nulls.index]
df_all= pd.get_dummies(df_all, drop_first=True)
fig, ax = plt.subplots(figsize=(12, 8))
corrmat = abs(df_all.corr())
cols = corrmat.nlargest(20, 'SalePrice')['SalePrice'].index
# cols = corrmat[corrmat['SalePrice'] > 0.5].index    # use this line or the line above
cm = abs(np.corrcoef(df_all[cols].dropna().values.T))

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
            yticklabels=cols.values, xticklabels=cols.values, ax=ax, cmap="YlGnBu")
ax.set_title('The highest correlated features with Sale Price')
plt.show();
# correlation with the target
top_corr = abs(df_all.corr())
top_features = top_corr.sort_values(by="SalePrice", ascending=False).head(20)
top_features['SalePrice'].sort_values( ascending=False)
df_all_xgb = df_all.copy()
# skewed_feat = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
skewed_feat = numeric_data.columns.to_list()
skewed_feat.append('SalePrice')
for feat in skewed_feat:
    df_all[feat] = np.log1p(df_all[feat])
# Train Data
newtraining = df_all.loc[:1460]
# Test Data
newtesting = df_all.loc[1461:].drop('SalePrice', axis=1)
newtraining.shape, newtesting.shape
newtraining_xgb = df_all_xgb.loc[:1460]
newtesting_xgb = df_all_xgb.loc[1461:].drop('SalePrice', axis=1)
# features_corr = abs(newtraining.corr()).sort_values(by='SalePrice', ascending=False).drop('SalePrice', axis=1).drop('SalePrice')

# # np.where will return 2 arrays the first one(the good one) contains row indexes,
# # and the second one contains columns indexes(to drop some)
# keep_feat, del_feat = np.where((features_corr > .90) & (features_corr < 1))
# # to get keep_feat and del_feat names 
# keep_feat = features_corr.iloc[keep_feat, :].index # use rows and .index to get columns names
# keep_feat = keep_feat.drop_duplicates()
# del_feat = features_corr.iloc[:, del_feat].columns # use columns and .columns to get columns names
# del_feat = del_feat.drop_duplicates()
# keep_and_del = list(zip(keep_feat, del_feat)) # Correlated columns
# keep_and_del
# feat_to_drop = []
# for col in keep_and_del:
#     if col[0] in del_feat:
#         if np.where(del_feat == col[0]) < np.where(keep_feat == col[0]):
#             feat_to_drop.append(col[0])
            
# feat_to_drop
# newtraining.drop(feat_to_drop, axis=1, inplace=True)
# newtesting.drop(feat_to_drop, axis=1, inplace=True)
# no_outliars_df.drop(feat_to_drop, axis=1, inplace=True)
y = newtraining['SalePrice']
X = newtraining.drop('SalePrice', axis=1)
y_xgb = newtraining_xgb['SalePrice']
X_xgb = newtraining_xgb.drop('SalePrice', axis=1)
print('Number of features before features selection: ', X.shape[1])
feat_sel = SelectFromModel(Lasso(alpha=0.001, random_state=42))
feat_sel.fit(X, y);
X = feat_sel.transform(X)
newtesting = feat_sel.transform(newtesting)
print('Number of features after features selection: ', X.shape[1])
# from sklearn.preprocessing import PowerTransformer
# pt = PowerTransformer(method='box-cox')
# Xpt = pt.fit_transform(X)
# tpt = pt.transform(newtesting)
ss = StandardScaler()
Xs =ss.fit_transform(X)
testing = ss.transform(newtesting)
X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=0.10, random_state=42)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=1, random_state=42)
def modeling(model, X_train, y_train, test_data, X_test=None, y_test=None, prefit=False):
    '''Takes model and data then print model results with some linear metrics then return predictions'''
    
    start = "\033[1m" # to create BOLD print
    end = "\033[0;0m" # to create BOLD print
    
    # Print bold model name 
    model_name = str(model).split('(')[0]
    print(''.join(['\n', start, model_name, end]))
    
    #Fit model
    if not prefit:
        model.fit(X_train, y_train)
    
    #Accuarcy score    
    print('Train Score', model.score(X_train, y_train))
    try:
        print('Test Score :', model.score(X_test, y_test))
    except: pass
    
    #cross val score
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -(cross_val_score(model, X_train, y_train, cv=cv, scoring = 'neg_mean_squared_error'))
    print('CV RMSE: ', np.sqrt(mse))
    print('CV RMSE mean: ', np.sqrt(mse).mean())
    
    #predictions
    y_pred = np.expm1(model.predict(test_data))
    print('\nFirst 5 Predictions: \n', y_pred[:5])  

    return y_pred
# Models
lr = LinearRegression()
lasso = Lasso(alpha=0.001)
ridge = Ridge(7)
elastic = ElasticNet(0.002335721469090121)
tree = DecisionTreeRegressor(max_depth=10, random_state=42,)
randomF = RandomForestRegressor(max_depth=18, random_state=42)
knn_reg = KNeighborsRegressor(n_neighbors=7)
lasso_cv = LassoCV(alphas= np.logspace(-5, 0, num=20), cv=5)
ridge_cv = RidgeCV(alphas= np.logspace(-5, 20, num=100), cv=5)
elastic_cv = ElasticNetCV(alphas= np.logspace(-5, 0, num=20), cv=5)



models = [(lr,'lr'), (lasso,'lasso'), (ridge,'ridge'), (elastic,'elastic'), (tree,'tree'), (randomF,'randomF'),
         (knn_reg,'knn_reg'), (lasso_cv,'lasso_cv'), (ridge_cv,'ridge_cv'), (elastic_cv,'elastic_cv')]

preds = {}    # empty dict to save all models predictions
for model, name in models:
    preds[name] = modeling(model, X_train, y_train, testing, X_test, y_test)
def g_search(model, param, X_train, y_train, test_data, X_test=None, y_test=None):
    '''Simple grid search with kfold'''
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(model,
                  param,
                  scoring='neg_mean_squared_error',
                  cv=cv,
                  n_jobs=-1,
                  verbose=0)
    gs.fit(X_train, y_train)
    
    # Results
    y_pred = modeling(gs.best_estimator_, X_train, y_train, test_data, X_test, y_test, prefit=True) # print results and return predictions
    
    print('Best parameters: ', gs.best_params_)
    
    return y_pred
# grid search using all the data (Xs, y)
grid_lasso_pred = g_search(Lasso(), {'alpha': np.logspace(-5, 0, num=20)}, Xs, y, testing)
grid_ridge_pred = g_search(Ridge(), {'alpha': np.arange(0, 400, 1)}, Xs, y, testing)
grid_elastic_pred = g_search(ElasticNet(), {'alpha': np.logspace(-5, 0, num=20)}, Xs, y, testing)
xgb_model = xgb.XGBRegressor(
                                 colsample_bytree=0.2,
                                 gamma=0.0,
                                 learning_rate=0.01,
                                 max_depth=4,
                                 min_child_weight=1.5,
                                 n_estimators=3800,                                                                  
                                 reg_alpha=0.9,
                                 reg_lambda=0.6,
                                 subsample=0.2,
                                 seed=42,
                                 tree_method='approx'
                                 )

xgb_hist = xgb_model.fit(X_train_xgb,y_train_xgb)
#                          eval_set=[(X_train_xgb,y_train_xgb),(X_test_xgb,y_test_xgb)],
#                          eval_metric='rmse',
#                          early_stopping_rounds=3000
#                         )
# before combining train and predicted fit with : n_estimators=3800
# after combining train and predicted fit with : n_estimators=4500
xgb_predictions = xgb_model.predict(newtesting_xgb)
xgb_predictions[:5]
cv=KFold(n_splits=5, shuffle=True, random_state=42)
mse = -(cross_val_score(xgb_model, X_xgb, y_xgb, cv=cv, scoring='neg_mean_squared_error'))
np.sqrt(mse).mean()
train_and_pred = pd.concat([X_xgb, newtesting_xgb])
new_y = y_xgb.append(pd.Series(xgb_predictions)).reset_index(drop=True)
new_y.index += 1
new_y.index.name = 'Id'
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    train_and_pred, new_y, test_size=1, random_state=42)
xgb_model.n_estimators = 4500  # change n_estimator from 3800 to 4500
xgb_model.fit(X_train_xgb,y_train_xgb)

xgb_predictions = xgb_model.predict(newtesting_xgb)
xgb_predictions[:5]
lgb_reg=lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.035,
                                n_estimators=3500, max_bin=50, bagging_fraction=0.65,bagging_freq=5, bagging_seed=7, 
                                feature_fraction=0.201, feature_fraction_seed=7,n_jobs=-1)
lgb_reg.fit(X_train, y_train)
print('Train: ',lgb_reg.score(X_train,y_train))
print('Test: ',lgb_reg.score(X_test,y_test))
cv=KFold(n_splits=5, shuffle=True, random_state=42)
mse = -(cross_val_score(lgb_reg, Xs, y, cv=cv, scoring='neg_mean_squared_error'))
np.sqrt(mse).mean()
lgb_reg_predictions = np.expm1(lgb_reg.predict(testing))
lgb_reg_predictions[:5]
new_pred = (xgb_predictions + grid_elastic_pred) / 2
new_pred[:5]
the_submission = submission.copy()
the_submission['SalePrice'] = new_pred
the_submission['SalePrice'].head()
the_submission.to_csv('the_submission.csv')
# Final Result: !!!!!!! KAGGLE SCORE: 0.11696 !!!!!!!

# This is the best result I got
# This result from (elastic_predictions + xgb_predictions) / 2

# And each model trained with different features and elastic with log1p and features selection but not xgboost
# ========================================== And finally ====================================

# new_pred = (elastic_predictions + xgb_predictions) / 2


# In the next two cells each model parameters 
# !!!! This resualt with adding the predicted values to the train again and fit with xgboost again
#                                        (the first fit: n_estimators=3800, the second fit: n_estimators=4500)
# xgb_model = xgb.XGBRegressor(
#                                  colsample_bytree=0.2,
#                                  gamma=0.0,
#                                  learning_rate=0.01,
#                                  max_depth=4,
#                                  min_child_weight=1.5,
#                                  n_estimators=3800 and 4500,                                                                  
#                                  reg_alpha=0.9,
#                                  reg_lambda=0.6,
#                                  subsample=0.2,
#                                  seed=42,
#                                  tree_method='approx'
#                                  )

# xgb_hist = xgb_model.fit(X_train,y_train,
#                          eval_set=[(X_train,y_train),(X_test,y_test)],
#                          eval_metric='rmse',
#                         )


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=1, random_state=42)    # test size only 1 because I want to train on almost all the data
# Kaggle score : 0.12047
# This result with np.log1p for all numeric features

# elastic=ElasticNet(0.001)
# elastic.fit(X, y)

# skewed_feat = numeric_data.columns.to_list()
# skewed_feat.append('SalePrice')
# for feat in skewed_feat:
#     df_all[feat] = np.log1p(df_all[feat])

# feat_sel = SelectFromModel(Lasso(alpha=0.001, random_state=42))
# feat_sel.fit(Xs, y)  # Standardized

# Kaggle score: 0.12339
# ============================================================ XGBRegressor ====================================
# xgb_model = xgb.XGBRegressor(
#                                  colsample_bytree=0.2,
#                                  gamma=0.0,
#                                  learning_rate=0.01,
#                                  max_depth=4,
#                                  min_child_weight=1.5,
#                                  n_estimators=3800,                                                                  
#                                  reg_alpha=0.9,
#                                  reg_lambda=0.6,
#                                  subsample=0.2,
#                                  seed=42,
#                                  tree_method='approx'
#                                  )

# xgb_hist = xgb_model.fit(X_train,y_train,
#                          eval_set=[(X_train,y_train),(X_test,y_test)],
#                          eval_metric='rmse',
#                         )

# X_train, X_test, y_train, y_test = train_test_split(
#     Xs, y, test_size=1, random_state=42)    # test size 1% only because we want to traim in almost all the data
# Kaggle score : 0.12102


# ============================================================ ElasticNet ====================================
# This result with np.log1p for all numeric features

# elastic=ElasticNet(0.001)
# elastic.fit(X, y)

# skewed_feat = numeric_data.columns.to_list()
# skewed_feat.append('SalePrice')
# for feat in skewed_feat:
#     df_all[feat] = np.log1p(df_all[feat])

# feat_sel = SelectFromModel(Lasso(alpha=0.001, random_state=42))
# feat_sel.fit(Xs, y)  # Standardized

# Kaggle score: 0.12339

# ============================================================ Lasso ====================================
# This result with np.log1p for all numeric features

# lasso=Lasso(0.0001)
# lasso.fit(X, y)

# skewed_feat = numeric_data.columns.to_list()
# skewed_feat.append('SalePrice')
# for feat in skewed_feat:
#     df_all[feat] = np.log1p(df_all[feat])


# feat_sel = SelectFromModel(Lasso(alpha=0.001, random_state=42))
# feat_sel.fit(X, y)


# score: 0.124475
# Kaggle score: 0.12352

# ============================================================ Ridge ====================================
# This result with np.log1p for all numeric features

# ridge = Ridge(2)
# ridge.fit(X, y)

# skewed_feat = numeric_data.columns.to_list()
# skewed_feat.append('SalePrice')
# for feat in skewed_feat:
#     df_all[feat] = np.log1p(df_all[feat])


# feat_sel = SelectFromModel(Lasso(alpha=0.001, random_state=42))
# feat_sel.fit(X, y)

# Kaggle score: 0.12390

# ============================================================ LGBMRegressor ====================================

# This resualt without filling numeric missing values (without the imputer)
# lgb_reg_o=lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.035,
#                                 n_estimators=2500, max_bin=50, bagging_fraction=0.65,bagging_freq=5, bagging_seed=7, 
#                                 feature_fraction=0.201, feature_fraction_seed=7,n_jobs=-1)
# lgb_reg_o.fit(Xo_train, yo_train)

# Xo_train, Xo_test, yo_train, yo_test = train_test_split(
#     Xso, yo, test_size=1, random_state=42)    # test size 1% only because we want to traim in almost all the data

# Xo_train, Xo_test, yo_train, yo_test = train_test_split(
#     Xso, yo, test_size=1, random_state=42)    # test size 1% only because we want to traim in almost all the data
# Kaggle score : 0.12595