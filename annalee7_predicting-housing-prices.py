import pandas as pd

import numpy as np



from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer



from scipy.stats import probplot

from scipy.stats import skew



import seaborn as sb

import matplotlib.pyplot as plt

from matplotlib import rcParams



%matplotlib inline

rcParams['figure.figsize'] = 8, 6

sb.set()
# Read in data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



print(train.shape)

print(test.shape)



train.head()
corr = train.corr()

plt.figure(figsize = (16, 12))

sb.heatmap(corr, linewidths = 0.5, fmt = '.2f', center = 1)

plt.show()
sb.pairplot(train[['SalePrice','OverallQual', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'GarageCars']])

plt.show()
y_train = train['SalePrice']

sb.distplot(y_train)

plt.xlabel('Sale Price')

plt.ylabel('Count')

plt.title('Distribution of Sale Price')

plt.show()



probplot(y_train, plot = plt)

plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])

y_train = train['SalePrice']

sb.distplot(y_train)

plt.xlabel('Sale Price')

plt.ylabel('Count')

plt.title('Distribution of Sale Price')

plt.show()



probplot(y_train, plot = plt)

plt.show()
plt.scatter(train['OverallQual'], train['SalePrice'])

plt.title('OveralQual vs. SalePrice')

plt.show()



plt.scatter(train['TotalBsmtSF'], train['SalePrice'])

plt.title('TotalBsmtSF vs. SalePrice')



plt.show()



plt.scatter(train['GrLivArea'], train['SalePrice'])

plt.title('GrLivArea vs. SalePrice')

plt.show()



plt.scatter(train['FullBath'], train['SalePrice'])

plt.title('FullBath vs. SalePrice')

plt.show()



plt.scatter(train['GarageCars'], train['SalePrice'])

plt.title('GarageCars vs. SalePrice')

plt.show()
train_features = train.drop(['SalePrice'], axis = 1)

test_features = test



features = pd.concat([train_features, test_features]).reset_index(drop = True)

features.shape
# Selecting numeric variables with skewness > 0.5

numvars = features.select_dtypes(include = ['int64', 'float64', 'int32']).columns

numvars_skew = pd.DataFrame(features[numvars].skew(), columns = ['Skew'])

numvars_skew = numvars_skew[numvars_skew['Skew'] > 0.5]



# Applying log transformation to skewed variables

skewed = features[numvars_skew.index]

unskewed = np.log1p(skewed)



# Replacing in dataset

features[skewed.columns] = unskewed
fig = plt.figure(figsize = (16, 12))

ax = fig.add_subplot()



sb.boxplot(data=features[skewed.columns] , orient="h")

plt.show()
# Sum of missing values

print("There are {} missing values in the features dataset.".format(features.isnull().sum().sum()))
# Summmary of columns with missing values

missing = features.columns[features.isnull().any()]

features[missing].isnull().sum().to_frame()
# Remove columns with more than 33 percent missing values: > 1000

remove_cols = features.columns[features.isnull().sum() > 1000]

print('The following list contains columns in the train dataset which have more than 33 percent missing values: \n{}.'.format(remove_cols))



# Drop those columns from features

features = features.drop(remove_cols, axis = 1)



print('I will remove those columns. \nThere are now {} missing values in the features dataset.'.format(features.isnull().sum().sum()))
# List of train columns that are objects

objs = features.select_dtypes(include = ['object']).columns



# List of train columns that are objects with missing values

missing = features[objs].columns[features[objs].isnull().any()]

features[missing].isnull().sum().to_frame()
# Fill train columns that are objects that have missing values with mode

features[missing] = features[missing].fillna(features.mode().iloc[0])

print("There are now {} missing values among the categorical variables.".format(features[objs].isnull().any().sum()))
features[objs] = features[objs].apply(LabelEncoder().fit_transform)

features.info()
missing = features.columns[features.isnull().any()]

features[missing].isnull().sum().to_frame()
imputer = SimpleImputer()

imputed_features = pd.DataFrame(imputer.fit_transform(features))

imputed_features.columns = features.columns

features = imputed_features



# Drop the ID columns

features = features.drop('Id', axis = 1)



features.head()
print("There are now {} missing values in the features dataset.".format(features.isnull().any().sum()))
features.shape
y_train = train['SalePrice']

X_train = features.iloc[:len(y_train), :]

X_test = features.iloc[len(y_train):, :]



X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



# ignore warnings

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning) 
seed = 1



linear = LinearRegression()

elastic = ElasticNet(random_state = seed)

ridge = Ridge(random_state = seed)

lasso = Lasso(random_state = seed)

kernel = KernelRidge()

r_forest = RandomForestRegressor(random_state = seed)

g_boost = GradientBoostingRegressor(random_state = seed)

svr = SVR()

knn = KNeighborsRegressor()

lgbm = LGBMRegressor(random_state = seed)

xgb = XGBRegressor(random_state = seed)
kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)



def cv_rmse(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = kfold))

    rmse = np.round(rmse, 6)

    return(rmse)
scores = {}

scores['linear'] = cv_rmse(linear).mean()

scores['elastic'] = cv_rmse(elastic).mean()

scores['ridge'] = cv_rmse(ridge).mean()

scores['lasso'] = cv_rmse(lasso).mean()

scores['kernel'] = cv_rmse(kernel).mean()

scores['r_forest'] = cv_rmse(r_forest).mean()

scores['g_boost'] = cv_rmse(g_boost).mean()

scores['svr'] = cv_rmse(svr).mean()

scores['knn'] = cv_rmse(knn).mean()

scores['lgbm'] = cv_rmse(lgbm).mean()

scores['xgb'] = cv_rmse(xgb).mean()
plt.scatter(scores.keys(), scores.values())

plt.xticks(rotation = 45)

plt.title('10-Fold Cross Validation Scores')

plt.ylabel('RMSE')

plt.xlabel('Model')

plt.show()
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
linear_model = linear.fit(X_train, y_train)

linear_pred = linear_model.predict(X_train)

linear_error = rmsle(y_train, linear_pred)



elastic_model = elastic.fit(X_train, y_train)

elastic_pred = elastic_model.predict(X_train)

elastic_error = rmsle(y_train, elastic_pred)



ridge_model = ridge.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_train)

ridge_error = rmsle(y_train, ridge_pred)



lasso_model = lasso.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_train)

lasso_error = rmsle(y_train, lasso_pred)



kernel_model = kernel.fit(X_train, y_train)

kernel_pred = kernel_model.predict(X_train)

kernel_error = rmsle(y_train, kernel_pred)



r_forest_model = r_forest.fit(X_train, y_train)

r_forest_pred = r_forest_model.predict(X_train)

r_forest_error = rmsle(y_train, r_forest_pred)



g_boost_model = g_boost.fit(X_train, y_train)

g_boost_pred = g_boost_model.predict(X_train)

g_boost_error = rmsle(y_train, g_boost_pred)



svr_model = svr.fit(X_train, y_train)

svr_pred = svr_model.predict(X_train)

svr_error = rmsle(y_train, svr_pred)



knn_model = knn.fit(X_train, y_train)

knn_pred = knn_model.predict(X_train)

knn_error = rmsle(y_train, knn_pred)



lgbm_model = lgbm.fit(X_train, y_train)

lgbm_pred = lgbm_model.predict(X_train)

lgbm_error = rmsle(y_train, lgbm_pred)



xgb_model = xgb.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_train)

xgb_error = rmsle(y_train, xgb_pred)
print("Linear Model RMSLE:", linear_error)

print("Elastic Net Model RMSLE:", elastic_error)

print("Ridge Model RMSLE:", ridge_error)

print("LASSO Model RMSLE:", lasso_error)

print("Kernel Ridge Model RMSLE:", kernel_error)

print("Random Forest Model RMSLE:", r_forest_error)

print("Gradient Boosting Model RMSLE:", g_boost_error)

print("Suppor Vector Regression Model RMSLE:", svr_error)

print("K-Nearest Neighbors Model RMSLE:", knn_error)

print("LightGBM RMSLE:", lgbm_error)

print("XGBoost Model RMSLE:", xgb_error)



model_names = ['linear', 'elastic', 'ridge', 'lasso', 'kernel', 'r_forest', 'g_boost', 'svr', 'knn', 'lgbm', 'xgb']

rmsle_scores = [linear_error, elastic_error, ridge_error, lasso_error, kernel_error, r_forest_error, 

                              g_boost_error, svr_error, knn_error, lgbm_error, xgb_error]



plt.scatter(model_names, rmsle_scores)

plt.xticks(rotation = 45)

plt.title('Fitted Model RMSLE Scores')

plt.ylabel('RMSLE')

plt.xlabel('Model')

plt.show()
blended_pred = (r_forest_pred*0.25 + g_boost_pred*0.15 + svr_pred*0.15 + lgbm_pred*0.30 + xgb_pred*0.15)

blended_error = rmsle(y_train, blended_pred)

print("Blended Model RMSLE:", blended_error)
model_names = ['linear', 'elastic', 'ridge', 'lasso', 'kernel', 'r_forest', 

               'g_boost', 'svr', 'knn', 'lgbm', 'xgb', 'blended']

rmsle_scores = [linear_error, elastic_error, ridge_error, lasso_error, kernel_error, r_forest_error, 

                              g_boost_error, svr_error, knn_error, lgbm_error, xgb_error, blended_error]

plt.scatter(model_names, rmsle_scores)

plt.xticks(rotation = 45)

plt.title('Fitted Model RMSLE Scores')

plt.ylabel('RMSLE')

plt.xlabel('Model')

plt.show()
r_forest_pred = r_forest_model.predict(X_test)

g_boost_pred = r_forest_model.predict(X_test)

svr_pred = r_forest_model.predict(X_test)

lgbm_pred = r_forest_model.predict(X_test)

xgb_pred = r_forest_model.predict(X_test)



blended_pred = (r_forest_pred*0.25 + g_boost_pred*0.15 + svr_pred*0.15 + lgbm_pred*0.30 + xgb_pred*0.15)



predictions = np.expm1(blended_pred)

submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = predictions

submission.to_csv('submission.csv', index = False)
submission.head()