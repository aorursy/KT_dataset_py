import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
#Data_train

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#Data_test

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#Data (Train + Test)

data = pd.concat((df_train, df_test)).reset_index(drop=True)

data.drop(['SalePrice'], axis=1, inplace=True)



df_train.head()
df_train.shape, df_test.shape
df_train.describe()
df_train.keys()
#Different types of the features

df_train.dtypes
#histogram

df_train['SalePrice'].hist(bins = 40)
#skewness & kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#correlation matrix

corrmat = df_train.corr()



#Plot a heatmap to visualize the correlations

f, ax = plt.subplots(figsize=(30, 19))

sns.set(font_scale=1.45)

sns.heatmap(corrmat, square=True,cmap='coolwarm');
correlations = corrmat["SalePrice"].sort_values(ascending=False)

features = correlations.index[0:10]

features
sns.pairplot(df_train[features], size = 2.5)

plt.show();
df_train.drop(['Id'], axis=1, inplace=True)

df_test.drop(['Id'], axis=1, inplace=True)

data.drop(['Id'], axis=1, inplace=True)
training_null = pd.isnull(df_train).sum()

testing_null = pd.isnull(df_test).sum()



null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null
#Based on the description data file provided, all the variables who have meaningfull Nan



null_with_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
#Replacing every Nan value with "None"



for i in null_with_meaning:

    df_train[i].fillna("None", inplace=True)

    df_test[i].fillna("None", inplace=True)

    data[i].fillna("None", inplace=True)
null_many = null[null.sum(axis=1) > 200]  #a lot of missing values

null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values

null_many
df_train.drop("LotFrontage", axis=1, inplace=True)

df_test.drop("LotFrontage", axis=1, inplace=True)

data.drop("LotFrontage", axis=1, inplace=True)
null_few
#I chose to use the mean function for replacement

df_train["GarageYrBlt"].fillna(df_train["GarageYrBlt"].mean(), inplace=True)

df_test["GarageYrBlt"].fillna(df_test["GarageYrBlt"].mean(), inplace=True)

data["GarageYrBlt"].fillna(data["GarageYrBlt"].mean(), inplace=True)

df_train["MasVnrArea"].fillna(df_train["MasVnrArea"].mean(), inplace=True)

df_test["MasVnrArea"].fillna(df_test["MasVnrArea"].mean(), inplace=True)

data["MasVnrArea"].fillna(data["MasVnrArea"].mean(), inplace=True)



df_train["MasVnrType"].fillna("None", inplace=True)

df_test["MasVnrType"].fillna("None", inplace=True)

data["MasVnrType"].fillna("None", inplace=True)
types_train = df_train.dtypes #type of each feature in data: int, float, object

num_train = types_train[(types_train == int) | (types_train == float)] #numerical values are either type int or float

cat_train = types_train[types_train == object] #categorical values are type object



#we do the same for the test set

types_test = df_test.dtypes

num_test = types_test[(types_test == int) | (types_test == float)]

cat_test = types_test[types_test == object]
#we should convert num_train and num_test to a list to make it easier to work with

numerical_values_train = list(num_train.index)

numerical_values_test = list(num_test.index)

fill_num = numerical_values_train+numerical_values_test



print(fill_num)
for i in fill_num:

    df_train[i].fillna(df_train[i].mean(), inplace=True)
fill_num.remove('SalePrice')
print(fill_num)
for i in fill_num:

    df_test[i].fillna(df_test[i].mean(), inplace=True)

    data[i].fillna(data[i].mean(), inplace=True)
df_train.shape, df_test.shape
categorical_values_train = list(cat_train.index)

categorical_values_test = list(cat_test.index)
fill_cat = []



for i in categorical_values_train:

    if i in list(null_few.index):

        fill_cat.append(i)

print(fill_cat)
def most_common_term(lst):

    lst = list(lst)

    return max(set(lst), key=lst.count)

#most_common_term finds the most common term in a series



most_common = []



for i in fill_cat:

    most_common.append(most_common_term(data[i]))

    

most_common
most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]],

                          fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]],

                          fill_cat[8]: [most_common[8]]}

most_common_dictionary
k = 0

for i in fill_cat:  

    df_train[i].fillna(most_common[k], inplace=True)

    df_test[i].fillna(most_common[k], inplace=True)

    data[i].fillna(most_common[k], inplace=True)

    k += 1
training_null = pd.isnull(df_train).sum()

testing_null = pd.isnull(df_test).sum()



null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])

null[null.sum(axis=1) > 0]
(np.log(df_train["SalePrice"])).hist(bins = 40)
df_train["LogPrice"] = np.log(df_train["SalePrice"])

df_train.head()
df_train_add = df_train.copy()



df_train_add['TotalSF']=df_train_add['TotalBsmtSF'] + df_train_add['1stFlrSF'] + df_train_add['2ndFlrSF']



df_train_add['Total_Bathrooms'] = (df_train_add['FullBath'] + (0.5 * df_train_add['HalfBath']) +

                               df_train_add['BsmtFullBath'] + (0.5 * df_train_add['BsmtHalfBath']))



df_train_add['Total_porch_sf'] = (df_train_add['OpenPorchSF'] + df_train_add['3SsnPorch'] +

                              df_train_add['EnclosedPorch'] + df_train_add['ScreenPorch'] +

                              df_train_add['WoodDeckSF'])



df_test_add = df_test.copy()



df_test_add['TotalSF']=df_test_add['TotalBsmtSF'] + df_test_add['1stFlrSF'] + df_test_add['2ndFlrSF']



df_test_add['Total_Bathrooms'] = (df_test_add['FullBath'] + (0.5 * df_test_add['HalfBath']) +

                               df_test_add['BsmtFullBath'] + (0.5 * df_test_add['BsmtHalfBath']))



df_test_add['Total_porch_sf'] = (df_test_add['OpenPorchSF'] + df_test_add['3SsnPorch'] +

                              df_test_add['EnclosedPorch'] + df_test_add['ScreenPorch'] +

                              df_test_add['WoodDeckSF'])
## For ex, if PoolArea = 0 , Then HasPool = 0 too



df_train_add['haspool'] = df_train_add['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_train_add['has2ndfloor'] = df_train_add['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_train_add['hasgarage'] = df_train_add['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_train_add['hasbsmt'] = df_train_add['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_train_add['hasfireplace'] = df_train_add['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



df_test_add['haspool'] = df_test_add['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_test_add['has2ndfloor'] = df_test_add['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_test_add['hasgarage'] = df_test_add['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_test_add['hasbsmt'] = df_test_add['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_test_add['hasfireplace'] = df_test_add['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df_train[df_train["SalePrice"] > 600000 ] #Discovering the outliers
categorical_values_train = list(cat_train.index)

categorical_values_test = list(cat_test.index)

print(categorical_values_train)
df_train_add = df_train.copy()

df_test_add = df_test.copy()

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

for i in categorical_values_train:

    df_train_add[i] = lb_make.fit_transform(df_train[i])

    

for i in categorical_values_test:

    df_test_add[i] = lb_make.fit_transform(df_test[i])
for i in categorical_values_train:

    feature_set = set(df_train[i])

    for j in feature_set:

        feature_list = list(feature_set)

        df_train.loc[df_train[i] == j, i] = feature_list.index(j)

        df_train_add.loc[df_train[i] == j, i] = feature_list.index(j)



for i in categorical_values_test:

    feature_set2 = set(df_test[i])

    for j in feature_set2:

        feature_list2 = list(feature_set2)

        df_test.loc[df_test[i] == j, i] = feature_list2.index(j)

        df_test_add.loc[df_test[i] == j, i] = feature_list2.index(j)
df_train_add.head()
df_test_add.head()
#Importing all the librairies we'll need



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, KFold
X_train = df_train_add.drop(["SalePrice","LogPrice"], axis=1)

y_train = df_train_add["LogPrice"]
from sklearn.model_selection import train_test_split #to create validation data set



X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_training,y_training)

print(lm)
# print the intercept

print(lm.intercept_)
print(lm.coef_)
predictions = lm.predict(X_valid)

predictions= predictions.reshape(-1,1)
submission_predictions = np.exp(predictions)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_valid, submission_predictions))

print('MSE:', metrics.mean_squared_error(y_valid, submission_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, submission_predictions)))
linreg = LinearRegression()

parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}

grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")

grid_linreg.fit(X_training, y_training)



print("Best LinReg Model: " + str(grid_linreg.best_estimator_))

print("Best Score: " + str(grid_linreg.best_score_))
linreg = grid_linreg.best_estimator_

linreg.fit(X_training, y_training)

lin_pred = linreg.predict(X_valid)

r2_lin = r2_score(y_valid, lin_pred)

rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))

print("R^2 Score: " + str(r2_lin))

print("RMSE Score: " + str(rmse_lin))
scores_lin = cross_val_score(linreg, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lin)))
ridge = Ridge()

parameters_ridge = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}

grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")

grid_ridge.fit(X_training, y_training)



print("Best Ridge Model: " + str(grid_ridge.best_estimator_))

print("Best Score: " + str(grid_ridge.best_score_))
ridge = grid_ridge.best_estimator_

ridge.fit(X_training, y_training)

ridge_pred = ridge.predict(X_valid)

r2_ridge = r2_score(y_valid, ridge_pred)

rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))

print("R^2 Score: " + str(r2_ridge))

print("RMSE Score: " + str(rmse_ridge))
scores_ridge = cross_val_score(ridge, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_ridge)))
from sklearn import ensemble
params = {'n_estimators': 20000, 'max_depth': 5, 'min_samples_split': 2,

          'learning_rate': 0.05, 'loss': 'ls' , 'max_features' : 20}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_training, y_training)
clf_pred=clf.predict(X_valid)

clf_pred= clf_pred.reshape(-1,1)

r2_clf = r2_score(y_valid, clf_pred)

rmse_clf = np.sqrt(mean_squared_error(y_valid, clf_pred))

print("R^2 Score: " + str(r2_clf))

print("RMSE Score: " + str(rmse_clf))
scores_clf = cross_val_score(clf, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_clf)))
from sklearn.tree import DecisionTreeRegressor

dtreg = DecisionTreeRegressor(random_state = 100)

parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 

                  "max_features" : ["auto", "log2"]}

grid_dtr = GridSearchCV(dtreg, parameters_dtr, verbose=1, scoring="r2")

grid_dtr.fit(X_training, y_training)



print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))

print("Best Score: " + str(grid_dtr.best_score_))
dtr = grid_dtr.best_estimator_

dtreg.fit(X_training, y_training)

dtr_pred = dtreg.predict(X_valid)

r2_dtr = r2_score(y_valid, dtr_pred)

rmse_dtr = np.sqrt(mean_squared_error(y_valid, dtr_pred))

print("R^2 Score: " + str(r2_dtr))

print("RMSE Score: " + str(rmse_dtr))
scores_dtr = cross_val_score(dtreg, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_dtr)))
rfr = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2"]}

grid_rf = GridSearchCV(rfr, paremeters_rf, verbose=1, scoring="r2")

grid_rf.fit(X_training, y_training)



print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))

print("Best Score: " + str(grid_rf.best_score_))
rf = grid_rf.best_estimator_

rfr.fit(X_training, y_training)

rf_pred = rfr.predict(X_valid)

r2_rf = r2_score(y_valid, rf_pred)

rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))

print("R^2 Score: " + str(r2_rf))

print("RMSE Score: " + str(rmse_rf))
scores_rf = cross_val_score(rfr, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_rf)))
from xgboost import XGBRegressor



xgboost = XGBRegressor(learning_rate=0.01,n_estimators=20000,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.006)

xgb = xgboost.fit(X_training, y_training)
xgb_pred = xgb.predict(X_valid)

r2_xgb = r2_score(y_valid, xgb_pred)

rmse_xgb = np.sqrt(mean_squared_error(y_valid, xgb_pred))

print("R^2 Score: " + str(r2_xgb))

print("RMSE Score: " + str(rmse_xgb))
from lightgbm import LGBMRegressor



lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=20000,

                                       max_bin=2000, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

gbm = lightgbm.fit(X_training, y_training)
gbm_pred = gbm.predict(X_valid)

r2_gbm = r2_score(y_valid, gbm_pred)

rmse_gbm = np.sqrt(mean_squared_error(y_valid, gbm_pred))

print("R^2 Score: " + str(r2_gbm))

print("RMSE Score: " + str(rmse_gbm))
model_performances = pd.DataFrame({

    "Model" : ["Linear Regression", "Ridge", "Decision Tree Regressor", "Random Forest Regressor","Gradient Boosting Regression","XGBoost","LGBM Regressor"],

    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5],  str(r2_dtr)[0:5], str(r2_rf)[0:5] , str(r2_clf)[0:5], str(r2_xgb)[0:5], str(r2_gbm)[0:5]],

    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8],  str(rmse_dtr)[0:8], str(rmse_rf)[0:8], str(rmse_clf)[0:8], str(rmse_xgb)[0:8], str(rmse_gbm)[0:8]]

})

model_performances.round(4)



print("Sorted by R Squared:")

model_performances.sort_values(by="R Squared", ascending=False)
print("Sorted by RMSE:")

model_performances.sort_values(by="RMSE", ascending=True)
learning_rates = [0.75 ,0.5, 0.25, 0.1, 0.05, 0.01]



r2_results = []

rmse_results = []



for eta in learning_rates:

    model = ensemble.GradientBoostingRegressor(learning_rate=eta)

    model.fit(X_training, y_training)

    y_pred = model.predict(X_valid)

    r2_clf = r2_score(y_valid, y_pred)

    rmse_clf = np.sqrt(mean_squared_error(y_valid, y_pred))

    r2_results.append(r2_clf)

    rmse_results.append(rmse_clf)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(learning_rates, r2_results, 'b', label='R^2')

line2, = plt.plot(learning_rates, rmse_results, 'r', label='RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Score')

plt.xlabel('learning_rates')

plt.show()
n_estimators = [1, 2, 16, 32, 64, 100, 200, 500]

r2_results = []

rmse_results = []



for estimator in n_estimators:

    model = ensemble.GradientBoostingRegressor(n_estimators=estimator)

    model.fit(X_training, y_training)

    y_pred = model.predict(X_valid)

    r2_clf = r2_score(y_valid, y_pred)

    rmse_clf = np.sqrt(mean_squared_error(y_valid, y_pred))

    r2_results.append(r2_clf)

    rmse_results.append(rmse_clf)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, r2_results, 'b', label='R^2')

line2, = plt.plot(n_estimators, rmse_results, 'r', label='RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Score')

plt.xlabel('n_estimators')

plt.show()
max_depths = np.linspace(1, 10, 10, endpoint=True)

r2_results = []

rmse_results = []



for max_depth in max_depths:

    model = ensemble.GradientBoostingRegressor(max_depth=max_depth)

    model.fit(X_training, y_training)

    y_pred = model.predict(X_valid)

    r2_clf = r2_score(y_valid, y_pred)

    rmse_clf = np.sqrt(mean_squared_error(y_valid, y_pred))

    r2_results.append(r2_clf)

    rmse_results.append(rmse_clf)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, r2_results, 'b', label='R^2')

line2, = plt.plot(max_depths, rmse_results, 'r', label='RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Score')

plt.xlabel('max_depths')

plt.show()
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

r2_results = []

rmse_results = []



for min_samples_split in min_samples_splits:

    model = ensemble.GradientBoostingRegressor(min_samples_split=min_samples_split)

    model.fit(X_training, y_training)

    y_pred = model.predict(X_valid)

    r2_clf = r2_score(y_valid, y_pred)

    rmse_clf = np.sqrt(mean_squared_error(y_valid, y_pred))

    r2_results.append(r2_clf)

    rmse_results.append(rmse_clf)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, r2_results, 'b', label='R^2')

line2, = plt.plot(min_samples_splits, rmse_results, 'r', label='RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Score')

plt.xlabel('min_samples_splits')

plt.show()
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

r2_results = []

rmse_results = []



for min_samples_leaf in min_samples_leafs:

    model = ensemble.GradientBoostingRegressor(min_samples_leaf=min_samples_leaf)

    model.fit(X_training, y_training)

    y_pred = model.predict(X_valid)

    r2_clf = r2_score(y_valid, y_pred)

    rmse_clf = np.sqrt(mean_squared_error(y_valid, y_pred))

    r2_results.append(r2_clf)

    rmse_results.append(rmse_clf)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leafs, r2_results, 'b', label='R^2')

line2, = plt.plot(min_samples_leafs, rmse_results, 'r', label='RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Score')

plt.xlabel('min_samples_leafs')

plt.show()
max_features = list(range(1,30))

r2_results = []

rmse_results = []



for max_feature in max_features:

    model = ensemble.GradientBoostingRegressor(max_features=max_feature)

    model.fit(X_training, y_training)

    y_pred = model.predict(X_valid)

    r2_clf = r2_score(y_valid, y_pred)

    rmse_clf = np.sqrt(mean_squared_error(y_valid, y_pred))

    r2_results.append(r2_clf)

    rmse_results.append(rmse_clf)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_features, r2_results, 'b', label='R^2')

line2, = plt.plot(max_features, rmse_results, 'r', label='RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Score')

plt.xlabel('max_features')

plt.show()
def blend_models_predict(X):

    return ((0.05 * lm.predict(X)) + \

            (0.05 * linreg.predict(X)) + \

            (0.05 * ridge.predict(X)) + \

            (0.1 * clf.predict(X)) + \

            (0.2 * gbm.predict(X)) + \

            (0.15 * rfr.predict(X)) + \

            (0.4 * xgb.predict(X)))

submission_predictions = np.exp(blend_models_predict(df_test_add))

print(submission_predictions)
res=pd.DataFrame(columns = ['Id', 'SalePrice'])

res['Id'] = df_test.index + 1461

res['SalePrice'] = submission_predictions

res.to_csv('submission1.csv',index=False)