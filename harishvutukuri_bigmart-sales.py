import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/Train.csv")
test = pd.read_csv("../input/Test.csv")
train.Item_Outlet_Sales.describe()
train.isna().sum()
train.Item_Fat_Content.unique()
train.Item_Fat_Content.replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'], inplace=True)
test['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
train["Item_Weight"] = train["Item_Weight"].fillna(train['Item_Weight'].dropna().mode().values[0])
train["Outlet_Size"] = train["Outlet_Size"].fillna(train['Outlet_Size'].dropna().mode().values[0])
test["Item_Weight"] = test["Item_Weight"].fillna(test['Item_Weight'].dropna().mode().values[0])
test["Outlet_Size"] = test["Outlet_Size"].fillna(test['Outlet_Size'].dropna().mode().values[0])
# Univariate graphs to see the distribution
train.hist(figsize=(20, 15))
plt.show()
# Correlation Matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(train.corr(), annot=True)
# Creating Dependent and Independent variable
X1 = train.drop(["Item_Identifier","Outlet_Identifier"],axis=1)
X2 = test.drop(["Item_Identifier","Outlet_Identifier"],axis=1)

X = X1.iloc[:,:-1]
y = X1.iloc[:,-1]
X_cv = X2

# Dummy variables
X = pd.get_dummies(X, drop_first=True)
X_cv = pd.get_dummies(X_cv, drop_first=True)
# Spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# feature scaling
from  sklearn.preprocessing  import StandardScaler

slc= StandardScaler()
X_train = slc.fit_transform(X_train)
X_cv = slc.transform(X_cv)
X_test = slc.transform(X_test)
# Test options and evaluation metric
num_folds = 10
seed = 0
scoring = 'neg_mean_squared_error'
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# Spot-Check Algorithms (Regression)
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Spot-Check Ensemble Models (Regression)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost.sklearn import XGBRegressor

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

models.append(('AB', AdaBoostRegressor()))
models.append(('GBM', GradientBoostingRegressor()))
models.append(('ET', ExtraTreesRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('XGB',XGBRegressor()))

# evaluate each model in turn
results = {}
rmse = {}
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results[name] = (cv_results.mean(), cv_results.std())
    model.fit(X_train, y_train)
    _ = model.predict(X_test)
    rmse[name] = np.sqrt(mean_squared_error(y_test, _))
results
rmse
# Parameter Tuning the Best Model from the results
from sklearn.model_selection import GridSearchCV

model = XGBRegressor(random_state=seed)

params = {'learning_rate':[0.1], 'n_estimators':[67], 'booster':['dart']}

kfold = KFold(n_splits=num_folds, random_state=seed)
grid_search = GridSearchCV(estimator = model ,param_grid = params,scoring=scoring ,cv =kfold, verbose = 4) 
grid_search.fit(X_train, y_train)
# Best Score and Best Parameters from GridSearch
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
# Finalizing the model and comparing the test, predict results

model = XGBRegressor(random_state=seed, n_estimators = 67, learning_rate=0.1, booster='dart')

_ = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
results["GScv"] = (_.mean(), _.std())

model.fit(X_train, y_train) 
y_predict = model.predict(X_test)

rmse["GScv"] = np.sqrt(mean_squared_error(y_test, y_predict))
print(r2_score(y_test, y_predict))
rmse
# Predicting
model = XGBRegressor(random_state=seed, n_estimators = 67, learning_rate=0.1, booster='dart')
model.fit(X_train, y_train) 

final_predict = model.predict(X_cv)
final_predict