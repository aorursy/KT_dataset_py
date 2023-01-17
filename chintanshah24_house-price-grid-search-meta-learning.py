import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
test_data = pd.read_csv('../input/test.csv')
X_test = test_data.copy(deep=True)
data.drop(data[(data["GrLivArea"]>4000)&(data["SalePrice"]<300000)].index,inplace=True)
X = data.drop('SalePrice', axis=1).drop('Id', axis=1)
ID = data['Id']
y = data['SalePrice']
test_ID = test_data['Id']
test_data.drop('Id', axis=1, inplace=True)
X.head()
train_size = X.shape[0]
all_data = pd.concat((X, test_data)).reset_index(drop=True)
ordinals = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
for ordinal in ordinals:
    all_data[ordinal] = all_data[ordinal].astype(str)
import numpy as np
qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, np.NaN:0}
all_data["ExterQual"] = all_data["ExterQual"].apply(lambda x: qual_dict[x])
all_data["ExterCond"] = all_data["ExterCond"].apply(lambda x: qual_dict[x])
all_data["BsmtQual"] = all_data["BsmtQual"].apply(lambda x: qual_dict[x])
all_data["BsmtCond"] = all_data["BsmtCond"].apply(lambda x: qual_dict[x])
all_data["HeatingQC"] = all_data["HeatingQC"].apply(lambda x: qual_dict[x])
all_data["KitchenQual"] = all_data["KitchenQual"].apply(lambda x: qual_dict[x])
all_data["FireplaceQu"] = all_data["FireplaceQu"].apply(lambda x: qual_dict[x])
all_data["GarageQual"] = all_data["GarageQual"].apply(lambda x: qual_dict[x])
all_data["GarageCond"] = all_data["GarageCond"].apply(lambda x: qual_dict[x])

all_data["BsmtExposure"] = all_data["BsmtExposure"].apply( lambda x:
    {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4, np.NaN: 0}[x])

bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, np.NaN:0}
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].apply(lambda x: bsmt_fin_dict[x])
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].apply(lambda x: bsmt_fin_dict[x])

all_data["Functional"] = all_data["Functional"].apply(lambda x:
    {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
     "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8, np.NaN: 0}[x])

all_data["GarageFinish"] = all_data["GarageFinish"].apply( lambda x:
    {None: 0, "Unf": 1, "RFn": 2, "Fin": 3, np.NaN: 0}[x])

all_data["Fence"] = all_data["Fence"].apply(
    lambda x: {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4, np.NaN: 0}[x])

all_data["YearBuilt"] = all_data["YearBuilt"]
all_data["YearRemodAdd"] = all_data["YearRemodAdd"]

all_data["GarageYrBlt"] = all_data["GarageYrBlt"]
all_data["GarageYrBlt"].fillna(0.0, inplace=True)

all_data["MoSold"] = all_data["MoSold"]
all_data["YrSold"] = all_data["YrSold"]

all_data["LowQualFinSF"] = all_data["LowQualFinSF"]
all_data["MiscVal"] = all_data["MiscVal"]

all_data["PoolQC"] = all_data["PoolQC"].apply(lambda x: qual_dict[x])

all_data["PoolArea"] = all_data["PoolArea"]
all_data["PoolArea"].fillna(0, inplace=True)
all_data.select_dtypes(exclude=['object']).head()
for ordinal in ['MSSubClass']: #Based on description
    all_data[ordinal] = all_data[ordinal].astype(str)
all_na = (all_data.isnull().sum() * 100)/ len(all_data)
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)
all_na
def fill_with_NA(df, column):
    df[column] = df[column].fillna("NA")
    return df
def fill_with_mode(df, column):
    df[column] = df[column].fillna(df[column].mode()[0])
    return df
all_data['PoolQC'].unique()
all_data = fill_with_NA(all_data, 'PoolQC')
all_data['MiscFeature'].unique()
all_data = fill_with_NA(all_data, 'MiscFeature') # Since majority is nan
all_data['Alley'].unique()
all_data = fill_with_NA(all_data, 'Alley') # Since majority is nan
all_data['Fence'].unique()
all_data = fill_with_NA(all_data, 'Fence') # Since majority is nan
all_data['FireplaceQu'].unique()
all_data = fill_with_NA(all_data, 'FireplaceQu') # Since majority is nan
plt.hist(all_data['LotFrontage'], range=(0, all_data['LotFrontage'].max()))
plt.show()
all_data["LotFrontage"] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
all_data.head()
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_data = fill_with_NA(all_data, col)
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    all_data[col]  = all_data[col].fillna(0)
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']:
    all_data[col] = all_data[col].fillna(0)
for col in ['BsmtFinType2','BsmtFinType1', 'BsmtExposure','BsmtCond','BsmtQual','Exterior2nd','Exterior1st','MasVnrType']:
    all_data = fill_with_NA(all_data, col)
all_data['MSZoning'].value_counts().plot(kind='bar')
all_data = fill_with_mode(all_data, 'MSZoning')
all_data['Functional'].value_counts().plot(kind='bar')
all_data = fill_with_mode(all_data, 'Functional')
all_data['Utilities'].value_counts().plot(kind='bar')
all_data = fill_with_mode(all_data, 'Utilities')
all_data['SaleType'].value_counts().plot(kind='bar')
all_data = fill_with_mode(all_data, 'SaleType')
all_data['KitchenQual'].value_counts().plot(kind='bar')
import seaborn as sns
sns.boxplot(x="KitchenQual", y="SalePrice", data=data)
all_data = fill_with_mode(all_data, 'KitchenQual')
all_data = fill_with_mode(all_data, 'Electrical')
all_na = (all_data.isnull().sum() * 100)/ len(all_data)
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)
all_na
all_data['ExterQual'] = all_data['ExterQual'].fillna("NA")
all_data['TotalAreaSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['AllSF'] = all_data['TotalAreaSF'] + all_data['WoodDeckSF'] + all_data['TotalBsmtSF'] +  all_data['OpenPorchSF']
import numpy as np
y_log = np.log1p(y)
from scipy.stats import norm, skew
sns.distplot(y_log , fit=norm)

print(skew(y_log))
numerical_columns = all_data.select_dtypes(exclude=['object']).columns
from scipy.special import boxcox1p
skewed = set()
log_skewed = set()
for column in numerical_columns:
    if abs(skew(all_data[column])) > 0.5: #Read how to select an ideal value for skew
        skewed.add(column)
        if abs(skew(np.log1p(all_data[column]))) > 0.5:
            log_skewed.add(column)
            
print("{} out of {} are skewed".format(len(skewed), len(numerical_columns)))
print("{} are not log normal".format(len(log_skewed)))
log_skewed
from scipy.special import boxcox1p
for column in (skewed):
#     all_data[column] = np.log1p(all_data[column])
    all_data[column] = boxcox1p(all_data[column], 0.15)
    # Since I don't yet know how to treat non log-normal data points
categorical_columns = all_data.select_dtypes(include=['object']).columns
from sklearn.preprocessing import LabelEncoder

for column in ['Alley',
 'CentralAir',
 'LandSlope',
 'LotShape',
 'MSSubClass',
 'MoSold',
 'OverallCond',
 'PavedDrive',
 'Street',
 'YrSold',
'YearBuilt','GarageYrBlt',
        ]:
    all_data[column] = LabelEncoder().fit_transform(all_data[column])
all_data.select_dtypes(include=['object']).columns
all_data = pd.get_dummies(all_data)
all_data.shape
train = all_data[:train_size]
test = all_data[train_size:]
from sklearn.model_selection import cross_val_score
def rms_error(model):
    return np.sqrt(-1 * cross_val_score(model, train.values, y_log.values, scoring="neg_mean_squared_error", cv = 5))
# from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor, HuberRegressor, LassoCV, RidgeCV, Lasso
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from dask_ml.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
pipeline = Pipeline([('model', RandomForestRegressor())])

random_forest_param_grid = {
    "model__n_estimators": [100,200, 300, 400, 500, 600, 700],
    "model__max_leaf_nodes": [ 100, 200, 300, 400, 500, 600, 650, 700, 800]
}

# CV set to 5
random_forest_grid = GridSearchCV(pipeline, cv=5, param_grid=random_forest_param_grid, n_jobs=-1)
random_forest_grid.fit(train, y_log)
random_forest_grid.best_params_
from dask_ml.model_selection import GridSearchCV
pipeline = Pipeline([('imputer',SimpleImputer()), ('model', GradientBoostingRegressor())])

gradient_boosting_param_grid = {
    "model__n_estimators": [ 500, 600,700,800,1000],
    "model__max_depth": [2, 3, 4]
}

# CV set to 5
gradient_boosting_grid = GridSearchCV(pipeline, cv=5, param_grid=gradient_boosting_param_grid, n_jobs=-1)
gradient_boosting_grid.fit(train, y_log)
gradient_boosting_grid.best_params_
from sklearn.model_selection import GridSearchCV
pipeline = Pipeline([('imputer',SimpleImputer()), ('model', XGBRegressor())])

param_grid = {
    "model__n_estimators": [500,600,800, 850, 1000, 1200],
    "model__learning_rate": [0.01, 0.02, 0.05,0.75, 0.1, 0.2]
}

fit_params = {"model__eval_set": [(train.as_matrix(), y_log.as_matrix())], 
              "model__early_stopping_rounds": 10, 
              "model__verbose": False} 
xgb_grid = GridSearchCV(pipeline, cv=5, param_grid=param_grid, fit_params = fit_params, n_jobs=32)
xgb_grid.fit(train, y_log)
xgb_grid.best_params_
pipeline_list = [
                Pipeline([('scaler', RobustScaler()), ('model', ElasticNetCV(alphas = [4, 1, 0.1, 0.001, 0.0005, 5e-4], l1_ratio = [1, 0.1, 0.001, 0.005]))]),
                Pipeline([('scaler', RobustScaler()), ('model', RidgeCV(alphas = [10, 1, 0.1, 0.001, 0.0005, 5e-4]))]),
                Pipeline([('scaler', RobustScaler()), ('model', LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4], max_iter=50000))]),
#                 Pipeline([('model', RandomForestRegressor(n_estimators=300, max_leaf_nodes=800))]),
                Pipeline([('model', XGBRegressor(learning_rate=0.05, n_estimators=800))]),
                Pipeline([('model', GradientBoostingRegressor(max_depth=2, n_estimators=600, loss='huber'))]), # Huber mentioned in https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
                Pipeline([('scaler', RobustScaler()), ('model', KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))]),
                Pipeline([('model', LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=600,
                              max_bin = 50, bagging_fraction = 0.6,
                              bagging_freq = 5, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11))])
                ]
model_names = ['elasticNet','ridge','lasso','XGB','GBR','KernelRidge','LGBM']
error_list =list()
for index, pipeline in enumerate(pipeline_list):
    error_list.append(rms_error(pipeline).mean())
errors = pd.DataFrame({
    'models': model_names,
    'error':error_list
})
errors.set_index('models', inplace=True)
errors
multiple = pd.DataFrame()
for index, pipeline in enumerate(pipeline_list):
    pipeline.fit(train, y_log)
    multiple[index] = pipeline.predict(test)
averaged_predictions = np.expm1(multiple.mean(axis=1))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

class StackingModel(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = max(len(base_models), 2)
    
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        k_folds = KFold(n_splits = self.n_folds, shuffle=True, random_state = 46)
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for ix, model in enumerate(self.base_models):
            for train_ix, test_ix in k_folds.split(X, y):
                m = clone(model)
                m.fit(X[train_ix], y[train_ix])
                self.base_models_[ix].append(m)
                y_predicted = m.predict(X[test_ix])
                out_of_fold_predictions[test_ix, ix] = y_predicted
        
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    def predict(self, X):
        return self.meta_model_.predict(
            np.column_stack(
                [
                    np.column_stack(model.predict(X) for model in models).mean(axis=1)
                    for models in self.base_models_
                ]
            )
        )
rms_error(StackingModel(base_models=pipeline_list, meta_model=Lasso(alpha =0.0005, random_state=1))).mean()
stack = StackingModel(base_models=pipeline_list, meta_model=Lasso(alpha =0.0005, random_state=1))
stack.fit(train.values, y_log.values)
m_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

m_xgb.fit(train.values, y_log.values)
m_lgbm = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
m_lgbm.fit(train.values, y_log.values)
answer = np.expm1(0.6 * stack.predict(test.values) + 0.2 * m_xgb.predict(test.values) + 0.2 * m_lgbm.predict(test.values))
submission = pd.DataFrame({'Id': test_ID, 'SalePrice':answer})
submission
submission.to_csv('submission_stacked.csv', index=False)