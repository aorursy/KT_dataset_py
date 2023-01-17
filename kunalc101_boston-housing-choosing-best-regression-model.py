import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
boston = load_boston()
print(boston.keys())
# Since, boston is a dictionary, let's check what it contains
print(boston.DESCR)
bos = pd.DataFrame(boston.data, columns=boston.feature_names)

bos['MEDV'] = boston.target
print("[INFO] bos df type : {}".format(type(bos)))
print("[INFO] bos df shape: {}".format(bos.shape))
print("[INFO] bos df features: {}".format(list(bos.columns.values)))
print("[INFO] bos df head():\n {}".format(bos.head()))
print(bos.describe())
# check for missing values in all the columns
print("[INFO] bos df isnull():\n {}".format(bos.isnull().sum()))
X = bos.drop('MEDV', axis=1)
Y = bos['MEDV']

print(X.shape)
print(Y.shape)
# splits the training and test data set in 75% : 25%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
# user variables to tune
seed = 5
folds = 10
metric = "neg_mean_squared_error"

# hold different regression models in a single dictionary
models = {}

# Linear algorithms
models['Linear'] = LinearRegression()
models['Lasso'] = Lasso()
models['ElasticNet'] = ElasticNet()
models['Ridge'] = Ridge()

# Nonlinear algorithms
models['KNN'] = KNeighborsRegressor()
models['SVR'] = SVR(gamma='auto')
models['DecisionTree'] = DecisionTreeRegressor()
models['BaggedDTree'] = BaggingRegressor(n_estimators=100, max_features=4)
models['RandomForest'] = RandomForestRegressor(n_estimators=100, max_features=4)
models['ExtraTrees'] = ExtraTreesRegressor(n_estimators=100, max_features=4)
models['AdaBoost'] = AdaBoostRegressor(n_estimators=100)
models['GradientBoost'] = GradientBoostingRegressor(n_estimators=100, max_features=4)
models['XGBoost'] = XGBRegressor(n_estimators=100, max_features=4)
models['LightGBM'] = LGBMRegressor(n_estimators=200)
# 10-fold cross validation for each model
model_results = []
model_names = []

for model_name in models:
    model = models[model_name]
    k_fold = KFold(n_splits=folds, random_state=seed)
    results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)
    
    model_results.append(results)
    model_names.append(model_name)
    print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))
# box-whisker plot to compare regression models
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Regression models comparison')

axis = fig.add_subplot(111)
plt.boxplot(model_results, showmeans=True, meanline=True)

axis.set_xticklabels(model_names, rotation=45, ha="right")
axis.set_ylabel("Mean Squared Error (MSE)")

plt.margins(0.05, 0.1)
plt.show()
# standardized the dataset
pipelines = {}

# Linear algorithms
pipelines['ScaledLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])
pipelines['ScaledLASSO'] = Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])
pipelines['ScaledElasticNet'] = Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])
pipelines['ScaledRidge'] = Pipeline([('Scaler', StandardScaler()), ('RIDGE', Ridge())])

# Nonlinear algorithms 
pipelines['ScaledKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])
pipelines['ScaledSVR'] = Pipeline([('Scaler', StandardScaler()), ('SVR', SVR(gamma='auto'))])
pipelines['ScaledDecisionTree'] = Pipeline([('Scaler', StandardScaler()), ('DecisionTree', DecisionTreeRegressor())])
pipelines["ScaledBaggedDTree"] = Pipeline([('Scaler', StandardScaler()), ('BaggedDTree', BaggingRegressor(n_estimators=100, max_features=4))])
pipelines["ScaledRandomForest"] = Pipeline([('Scaler', StandardScaler()), ('RandomForest', RandomForestRegressor(n_estimators=100, max_features=4))])
pipelines["ScaledExtraTrees"] = Pipeline([('Scaler', StandardScaler()), ('ExtraTrees', ExtraTreesRegressor(n_estimators=100, max_features=4))])
pipelines["ScaledAdaBoost"] = Pipeline([('Scaler', StandardScaler()), ('AdaBoost', AdaBoostRegressor(n_estimators=100))])
pipelines["ScaledGradientBoost"] = Pipeline([('Scaler', StandardScaler()), ('GradientBoost', GradientBoostingRegressor(n_estimators=100, max_features=4))])
pipelines["ScaledXGBoost"] = Pipeline([('Scaler', StandardScaler()), ('XGBoost', XGBRegressor(n_estimators=100, max_features=4))])
pipelines["ScaledLightGBM"] = Pipeline([('Scaler', StandardScaler()), ('LightGBM', LGBMRegressor(n_estimators=200))])
# 10-fold cross validation for each model
pipeline_results = []
pipeline_names = []

for pipeline_name, model in pipelines.items():
    k_fold = KFold(n_splits=folds, random_state=seed)
    results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)
    
    pipeline_results.append(results)
    pipeline_names.append(pipeline_name)
    print("{}: {}, {}".format(pipeline_name, round(results.mean(), 3), round(results.std(), 3)))
# box-whisker plot to compare regression models
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Scaled Regression models comparison')

axis = fig.add_subplot(111)
plt.boxplot(pipeline_results, showmeans=True, meanline=True)

axis.set_xticklabels(pipeline_names, rotation=45, ha="right")
axis.set_ylabel("Mean Squared Error (MSE)")

plt.margins(0.05, 0.1)
plt.show()
rf = RandomForestRegressor(random_state=seed)  

# X_train = StandardScaler().fit_transform(X_train)
# X_test = StandardScaler().fit_transform(X_test)

param_grid={
            'n_estimators': range(100, 500, 50),
            'max_features': range(2, 5, 1)   
            }

gsc = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='r2', cv=5)

grid_result = gsc.fit(X_train, Y_train)

print("\nGridSearch result: Best r2: %.3f using %s" % (grid_result.best_score_, grid_result.best_params_))

#create and fit the best regression model
best_model = RandomForestRegressor(**grid_result.best_params_)

best_model.fit(X_train, Y_train)

#make predictions using the model
Y_test_predict = best_model.predict(X_test)

print("\nThe RandomForest model performance for test set")
print("RandomForest MAE : %.3f" % mean_absolute_error(Y_test, Y_test_predict))
print("RandomForest MSE : %.3f" % mean_squared_error(Y_test, Y_test_predict))
print("RandomForest RMSE : %.3f" % np.sqrt(mean_squared_error(Y_test, Y_test_predict)))
print("RandomForest R2 score is : %.3f" % r2_score(Y_test, Y_test_predict))
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Actual vs Predicted MEDV [RandomForest]')
plt.scatter(Y_test, Y_test_predict, alpha=0.4)
plt.plot([0, 50], [0, 50], '--k')
 
plt.xlabel('True price MEDV ($1000s)')
plt.ylabel('Predicted price MEDV ($1000s)')
 
plt.show()