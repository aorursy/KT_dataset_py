%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_description = '../input/home-data-for-ml-course/data_description.txt'
file_path = '../input/home-data-for-ml-course/train.csv'
data = pd.read_csv(file_path)

data
data.info()
data.columns
data.dtypes
data.drop(columns='Id', inplace=True)
numerical_attr = data.select_dtypes(exclude=['object']).columns
numerical_attr
categorical_attr = data.select_dtypes(include=['object']).columns
categorical_attr
data.describe(include=np.number)
features = ['LotArea', 'OverallQual', 'OverallCond', 'LowQualFinSF','GrLivArea','SalePrice']

data.hist(column=features, figsize=(20,15))
corr_matrix = data.corr()

corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
features = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF', '1stFlrSF']

scatter_matrix(data[features], figsize=(20, 15))
data.plot(kind="scatter", x="1stFlrSF", y="SalePrice", alpha=0.5)
data.plot(kind="scatter", x="TotalBsmtSF", y="SalePrice", alpha=0.5)
data.plot(kind="scatter", x="GrLivArea", y="SalePrice", alpha=0.5)
import seaborn as sns

var = 'SaleType'

sns.boxplot(x=var, y='SalePrice', data=data)
var = 'SaleCondition'

sns.boxplot(x=var, y='SalePrice', data=data)
plt.xticks(rotation=70)
var = 'MSSubClass'

sns.boxplot(x=var, y='SalePrice', data=data)
ms_sub_class = data['MSSubClass'].unique()

ms_sub_class.sort()
ms_sub_class
var = 'MSZoning'

sns.boxplot(x=var, y='SalePrice', data=data)
var = 'BldgType'

sns.boxplot(x=var, y='SalePrice', data=data)
#descriptive statistics summary
data['SalePrice'].describe()
#histogram
sns.distplot(data['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % data['SalePrice'].skew())
print("Kurtosis: %f" % data['SalePrice'].kurt())
y = data['SalePrice']
data = data.drop(['SalePrice'], axis=1)
data
c_features = ['SaleType', 'SaleCondition', 'MSSubClass', 'MSZoning', 'BldgType']

selected_features = c_features + features

def get_df_missing_info(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = ((df.isnull().sum()/df.isnull().count()) * 100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
    return missing_data
    
missing_data = get_df_missing_info(data)
missing_data.loc[numerical_attr, :]
missing_data.loc[categorical_attr, :]
col_to_drop = ['Alley','PoolQC','Fence','MiscFeature']
for col in col_to_drop:
    categorical_attr = categorical_attr.drop(col)
data.drop(columns=col_to_drop, inplace=True)
from sklearn.pipeline import Pipeline 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


num_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer()),
    ('std_scaler', StandardScaler())
])
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer(
    transformers=[
    ("num", num_pipeline, list(numerical_attr)),
    ("cat", categorical_pipeline, list(categorical_attr))
])

final_columns = data.columns

data_prepared = full_pipeline.fit_transform(data)
data_prepared.shape
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, ElasticNet
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
X = data_prepared
lin_reg = LinearRegression(n_jobs=-1)
scores = cross_val_score(lin_reg, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
ridge = Ridge(alpha=0.1)
scores = cross_val_score(ridge, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
lasso = Lasso(alpha=0.1, max_iter=2000, tol=1e-2)
scores = cross_val_score(lasso, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
sdg = SGDRegressor(eta0=0.01)
scores = cross_val_score(sdg, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
ela = ElasticNet(alpha=0.1, l1_ratio=0.5)
scores = cross_val_score(ela, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.linear_model import BayesianRidge
brd = BayesianRidge()
scores = cross_val_score(brd, X.toarray(), y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.linear_model import ARDRegression

ard = ARDRegression()
scores = cross_val_score(ard, X.toarray(), y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.tree import DecisionTreeRegressor

tree_clf = DecisionTreeRegressor()
scores = cross_val_score(tree_clf, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=1)
scores = cross_val_score(rf, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.ensemble import ExtraTreesRegressor

ext_reg = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=1)
scores = cross_val_score(ext_reg, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.ensemble import AdaBoostRegressor

ada  = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=200, learning_rate=0.5, n_jobs=-1)

scores = cross_val_score(ada, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(n_estimators=120, criterion='mse', n_jobs=-1)

scores = cross_val_score(gbrt, X, y, scoring="neg_mean_squared_error", cv=10)
scores = np.sqrt(-scores)
display_scores(scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'tol':np.arange(start=1e-3, stop=11e-3, step=1e-3)}
]

ard = ARDRegression()

ard_grid_search = GridSearchCV(ard, param_grid, cv=10, 
                          scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)

ard_grid_search.fit(X.toarray(), y)

ard_grid_search.best_params_
ard_grid_search.best_estimator_
cvres = ard_grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
from sklearn.model_selection import RandomizedSearchCV 


param_grid =  {
    'n_estimators': [100,500,1000], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}

ext_reg = ExtraTreesRegressor(random_state=1)

ext_rndm_search = RandomizedSearchCV(ext_reg, param_grid, cv=10, 
                          scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)

ext_rndm_search.fit(X, y)

ext_rndm_search.best_params_
ext_rndm_search.best_estimator_
cvres = ext_rndm_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
gbrt = GradientBoostingRegressor(random_state=1)

param_grid = {
    'n_estimators' : [120, 220, 320, 420, 520],
    'criterion' : ['friedman_mse', 'mse', 'mae'],
    'loss' : ['ls', 'lad', 'huber', 'quantile'],
    'learning_rate' : [.01,0.1,0.2,0.3,0.5, 0.7, 0.9],
    'max_depth': [2, 5, 10, 15, 20, 25, None],
    'subsample': [0.5,0.6,0.7, 0.8, 0.9],
    'max_features': ['auto', 'sqrt', 'log2'],
}

gbrt_rndm_search = RandomizedSearchCV(gbrt, param_grid, cv=10, 
                          scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)

gbrt_rndm_search.fit(X, y)

gbrt_rndm_search.best_params_
gbrt_rndm_search.best_estimator_
cvres = gbrt_rndm_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
test
test = test[final_columns]
test_prepared = full_pipeline.transform(test)

test_prepared.shape
final_ard = ard_grid_search.best_estimator_

final_ard_predictions = final_ard.predict(test_prepared)

ids = np.arange(start=1461, stop=2920, step=1)

final_ard_predictions = pd.DataFrame({'Id': ids, 'SalePrice': final_ard_predictions})

final_ard_predictions
final_ext = ext_rndm_search.best_estimator_

final_ext_predictions = final_ext.predict(test_prepared)

final_ext_predictions = pd.DataFrame({'Id': ids, 'SalePrice': final_ext_predictions})

final_ext_predictions
final_gbrt = gbrt_rndm_search.best_estimator_

final_gbrt_predictions = final_gbrt.predict(test_prepared)

final_gbrt_predictions = pd.DataFrame({'Id': ids, 'SalePrice': final_gbrt_predictions})

final_gbrt_predictions
final_ard_predictions.to_csv('./ard_submission.csv', index=False)
final_ext_predictions.to_csv('./ext_submission.csv', index=False)
final_gbrt_predictions.to_csv('./gbrt_submission.csv', index=False)