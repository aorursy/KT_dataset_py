!pip install dabl
## Intial Setup.
# for basic operations
import numpy as np 
import pandas as pd 
# Machine Learning Tools
import sklearn as sk
import statsmodels.api as sm
import scipy as sp
# for data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
%matplotlib inline

# for getting the file path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# for avoiding warnings
import warnings
warnings.filterwarnings('ignore')
# reading the data
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# combine train and test data set
data = pd.concat([train, test],ignore_index=True, sort=False)
# Drop the index column
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
# lets check the head of the dataset
pd.set_option('max_columns', 82)
train
# Useful package Dabl for initial Explanatory Analysis.
plt.rcParams['figure.figsize'] = (15, 6)
plt.style.use('fivethirtyeight')
# Dabl is a useful python package that provides general overview of the data
import dabl
dabl.plot(train, target_col = 'SalePrice')
fig, ax = plt.subplots(1,2)
fig.set_size_inches(15, 6)
sns.distplot(train["SalePrice"], ax=ax[0])
sns.boxplot(train["SalePrice"], ax=ax[1])
# Try log transformation to correct for skewness
train["SalePrice"] = np.log(train["SalePrice"])
sns.distplot(train['SalePrice'])
plt.show()
# Check Feature Corrleations.
cor = train.corr()
plt.figure(figsize=(15,10))
sns.heatmap(cor, cmap="YlGnBu", vmax=0.9, cbar=False, square=True, linewidths=0.1)
plt.show()
# Find the top correlated covariates to the response variable
cor[abs(cor["SalePrice"].values) >= 0.5]["SalePrice"].sort_values(ascending=False)[1:]
# Feature: OverallQual
fig = plt.figure(constrained_layout=True, figsize=(8,5))
sns.boxplot(train["OverallQual"], train["SalePrice"])
plt.show()
# Feature GrLivArea
fig = plt.figure(constrained_layout=True, figsize=(8,5))
sns.scatterplot(x = train["GrLivArea"], y = train["SalePrice"])
plt.show()
# Remove Outlier based on GrLivArea
train = train.drop(train[(train["GrLivArea"]>4000) & (train["SalePrice"] < 12.5)].index)
train.reset_index(drop=True, inplace=True)
# More visualization of correlation between variables
style.use('ggplot')
sns.set_style('dark')
sns.set_context("paper")
plt.subplots(figsize=(30,20))

mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(train.corr(), cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9), mask=mask, annot=True, center=0)
plt.title("Heatmap of all Features", fontsize=20);
# We must correct for all NaN or features of the data
data = pd.concat((train, test)).reset_index(drop=True)
# Find NaN rows
nan_total = data.isnull().sum().sort_values(ascending=False)
nan_percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)
nan_data = pd.concat([nan_total, nan_percent], axis=1, keys=["Total", "Percent"])
nan_data[nan_data.Percent != 0]
# Separation of Categorical and numerical features
data_categorical = data.select_dtypes("object")
data_numerical = data.select_dtypes(["int64", "float64"])
# Fill NaN with None if the count indicates the number of facility in the house
count_col1 = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',
       'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure',
       'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
# Assume that these NaN values have standard features 
count_col2 = ["Functional", 'Exterior2nd', 'Exterior1st']
count_col3 = ["MSZoning", "Electrical", "KitchenQual","SaleType"]
for col in count_col1:
    data_categorical[col].fillna('None',inplace =True)
for col in count_col2:
    data_categorical[col].fillna(data[col].value_counts().idxmax(), inplace = True)
for col in count_col3:
    data_categorical[col].fillna(data[col].mode()[0], inplace=True)
# Fill NaN with 0 for conti. values
cont_col1 = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFullBath',
       'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea', 'GarageCars',
       'BsmtFinSF2', 'BsmtFinSF1',"MSSubClass"]
for col in cont_col1:
    data_numerical[col].fillna(0, inplace=True)
# Some numerical feature should be turned categorical
df = pd.concat([data_categorical, data_numerical], axis=1)
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['GarageYrBlt'] = df['GarageYrBlt'].astype(str)
df.drop('SalePrice', axis=1, inplace = True)
df.drop("Utilities", axis=1, inplace=True)
# Addition of important data
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df['2ndFlrSF']
# Check for any more missing Data
na_data = (df.isnull().sum() / len(df)) * 100
na_data = na_data.drop(na_data[na_data==0].index).sort_values(ascending=False)
ms_data = pd.DataFrame({"Missing Ratio" :na_data})
ms_data
# Filter for skewness in the covariates
numeric_feats = df.dtypes[df.dtypes!="object"].index
skewed_feats = df[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_feats
# Apply Box-cox Transformation for skewed features
high_skewed = skewed_feats[skewed_feats > 0.75]
skew_index = high_skewed.index
for i in skew_index:
    df[i] = sp.special.boxcox1p(df[i], sp.stats.boxcox_normmax(df[i]+1))
# One_hot encoding of the categorical variables
final_features = pd.get_dummies(df).reset_index(drop=True)
final_features.head()
print("Features size:", df.shape)
print("Final Features size:", final_features.shape)
nrow_train = train.shape[0]
X_train = final_features[:nrow_train]
X_test = final_features[nrow_train:]
y_train = train["SalePrice"]
from sklearn.feature_selection import SelectPercentile, f_regression
# Select Percentile selects features according to a percentile of the highest scores
# Weare using f_regression: F-value between label/feature for regression tasks
sel_f = SelectPercentile(f_regression, percentile=20)
x_best = sel_f.fit_transform(X_train, y_train)
# The support is the integer index of the selected features
support = np.asarray(sel_f.get_support())
pd.set_option('display.float_format', '{:.2e}'.format)

features = np.asarray(X_train.columns.values)
features_with_support = features[support]
# Get the calculated fscores
fscores = np.asarray(sel_f.scores_)
fscores_with_support = fscores[support]
# Get the calculated pvalues
pvalues = np.asarray(sel_f.pvalues_)
pvalues_with_support = pvalues[support]

top20 = pd.DataFrame({"F-score" :fscores_with_support,
                     "p-values" :pvalues_with_support},
                    index = features_with_support)
# top20.index.name = 'Feature'
print('Top 20% best associated features to SalePrice\nNumber of features:',len(features_with_support))
print(top20.sort_values(by = 'p-values', ascending = 'True'))
best_feat = X_train[features_with_support]
# Initially divide the training set to check for best model.
X_temp, X_valid, y_temp, y_valid = sk.model_selection.train_test_split(X_train, y_train,
                                                                         test_size=0.1, random_state=42)
print(X_temp.shape)
print(X_valid.shape)
print(y_temp.shape)
print(y_valid.shape)
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score

# Set the super params.
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds))
# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
elastic_model = elasticnet.fit(X_temp, y_temp)
lasso_model = lasso.fit(X_temp, y_temp)
ridge_model = ridge.fit(X_temp, y_temp)

elastic_pred = elastic_model.predict(X_valid)
lasso_pred = lasso_model.predict(X_valid)
ridge_pred = ridge_model.predict(X_valid)
reg_blend = ((elastic_model.predict(X_valid)) + (lasso_model.predict(X_valid)) + \
               (ridge_model.predict(X_valid)))/3
print("RMSE Elastic: ",np.sqrt(sk.metrics.mean_squared_error(np.exp(elastic_pred), y_valid)))
print("RMSE Lasso: ",np.sqrt(sk.metrics.mean_squared_error(np.exp(lasso_pred), y_valid)))
print("RMSE Ridge: ",np.sqrt(sk.metrics.mean_squared_error(np.exp(ridge_pred), y_valid)))
print("RMSE Blend: ", np.sqrt(sk.metrics.mean_squared_error(np.exp(reg_blend), y_valid)))
# Choose a model and predict using the test data set for submission
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.expm1(elastic_model.predict(X_test))

#Fix Outlier Predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression1.csv", index=False)
print(submission)
lasso_col = pd.DataFrame(data = lasso_model.named_steps["lassocv"].coef_, index = X_temp.columns, columns = ["Lasso Coefs"])
#lasso_col.sort_values(by = "Lasso Coefs", ascending=False)
list(lasso_col[lasso_col.iloc[:,0] != 0].index)
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use the best selected feature from before feature engineering section
y_train = train["SalePrice"]

train_data = X_train[features_with_support]
test_data = X_test[features_with_support]

train_dmat = xgb.DMatrix(data = train_data, label=y_train)
test_dmat = xgb.DMatrix(data = test_data)

print(train_data.shape)
print(y_train.shape)
params = {
    # Parameters that we are going to tune.
    'max_depth':6, #Maximum depth for a single tree in the forest
    'min_child_weight': 1, # Weight for creating a new child
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:squarederror', #Loss function to optimize for
    'eval_metric': "mae" #Criteria for evaluating the model
}

num_boost_round = 999 #Set early_stopping-round, so the model won't be trained this long
# Example of using XGBRegressor
data_dmatrix = xgb.DMatrix(data=train_data, label=y_train)
X_temp, X_test, y_temp, y_test = sk.model_selection.train_test_split(train_data, y_train,
                                                                       test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 5, alpha = 2, n_estimators = 20)
xg_reg.fit(X_temp, y_temp)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, preds))
print("RMSE: %f " % (rmse))
xgb.to_graphviz(xg_reg, num_trees=10)

params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 2}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=10,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results
"""
Function to grid search and find the best combination of params for a random forest model
params: The total of original parameter to start from
target_param: Target parameter to grid search for support as many dimension as possible
target_param_range: The ranges for each of the target parameters, the index should be matching with the 'target_param'
crit: the criteria to compare the param effectiveness (eg. "mae", "rmse")
# "train_dmat" must be set before,
"""

grid_param = {"max_depth": list(range(5,12)), "min_child_weight": list(range(4,10))}

def rf_grid_search(params, grid_param, crit="rmse"):
    target_param = grid_param.keys()
    target_param_range = grid_param.values()
    from itertools import product
    min_err = float("Inf")
    best_params = None
    for combi in product(*target_param_range):
        param_info = ["{}={}".format(str(x),str(y)) for x, y in zip(target_param, combi)]
        print("CV with {}".format(", ".join(param_info)))
        # Set the crossvalidation params here
        cv_results = xgb.cv(params, train_dmat, num_boost_round = num_boost_round,
                           seed=111, nfold=10, metrics={crit}, early_stopping_rounds =10)

        mean_err = cv_results['test-{}-mean'.format(crit)].min()
        boost_rounds = cv_results['test-{}-mean'.format(crit)].argmin()
        
        print("\t {}: {} for {} rounds".format(crit, mean_err, boost_rounds))
        if mean_err < min_err:
            min_err = mean_err
            best_params = combi

    print("Grid Search Results:")
    print("\t Best Params: {} with error - {}".format(best_params, min_err))
        
rf_grid_search(params, grid_param)
# Optimum Params using MAE
# Using gridsearch function
params["max_depth"] = 5
params["min_child_weight"] = 6
params['subsample'] = 0.8
params['colsample_bytree'] = 1.
params['eta'] = 0.01
dtrain = xgb.DMatrix(X_temp, label=y_temp)
dtest = xgb.DMatrix(X_test, label=y_test)
xgbreg_model = xgb.train(params, dtrain, num_boost_round = num_boost_round,
                 evals=[(dtest, "Test")],
                 early_stopping_rounds=10)
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xbg
import lightgbm as lgb
#Validation function
n_folds = 5
train = X_train[features_with_support]

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
# Establish Base Models
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
Enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                  max_depth=5, max_features='sqrt',
                                  min_samples_leaf=15, min_samples_split=10,
                                  loss="huber", random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4503, gamma=0.0458,
                            learning_rate=0.05, max_depth=3,
                            min_child_weight=1.7817, n_estimators=2200,
                            reg_alpha=0.4640, reg_lambda=0.8571,
                            subsample=0.5213, silent=1,
                            random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective="regression", num_leaves=5,
                             learning_rate=0.05, n_estimators=720,
                             max_bin=55, bagging_fraction=0.8,
                             bagging_freq=5, feature_fraction=0.2319,
                             feature_fraction_seed=9, bagging_seed=9,
                             min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

xtemp = X_train[features_with_support]
xtest = X_test[features_with_support]
models = [lasso, Enet, KRR, GBoost, model_xgb, model_lgb]
# We train the model
for m in models:
    score = rmsle_cv(m)
    print("\Model Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# Average Stacking
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (lasso, lasso))
score = rmsle_cv(averaged_models)
print(score)
print("Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Convinient stacking using the sklearn StackingRegressor
estimators = [
    ('GBoose', GBoost),
    ('ENet', Enet),
    ('KRR', KRR),
    ('xgb', model_xgb),
    ('lbg', model_lgb),
]
stacked_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=lasso
)
# calculate the score of the stacked regression
score = rmsle_cv(stacked_reg)
print(score)
print("Stacking models average score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# Choose a model and predict using the test data set for submission
Dmat_finaltest = xgb.DMatrix(X_test[features_with_support])

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = model.predict(Dmat_finaltest)

#Fix Outlier Predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)
print(submission)