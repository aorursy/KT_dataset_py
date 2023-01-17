import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter('ignore')
#df = pd.read_fwf("housing.data",header=None)
from sklearn.datasets import load_boston
boston_data=load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df['MEDV'] = boston_data.target
df.head()
#df['CHAS'].sum()
#df = df.drop(df.columns[[0,1,2,3,4,6,7,8,9,11]], axis=1)
df.describe()
df.info()
plt.figure(figsize=(20,5))
sns.heatmap(df.isnull())
plt.title("Missing Value Visualization")
plt.figure(figsize=(12,10))
sns.distplot(df['MEDV'],bins=30)
plt.xlabel("Median value of owner-occupied homes in 1000's of dollars")
plt.ylabel("Frequency")
plt.title("Target Variable (MEDV) Distribution")
plt.show()
from scipy import stats 
from scipy.stats import skew
from scipy.stats import norm

fig = plt.figure(figsize=(12,8))
res = stats.probplot(df['MEDV'],plot=plt)
plt.show()

print("Skewness: %f" % df['MEDV'].skew())
print("Kurtosis: %f" % df['MEDV'].kurt())
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
df.hist(ax = ax, layout=(4,4), sharex=False)
plt.tight_layout()
plt.show()
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
df.plot(ax=ax, kind='density', subplots=True, layout=(4,4), sharex=False)
plt.tight_layout()
plt.show()
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
df.plot(ax=ax, kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()
sns.set(style="ticks", font_scale=2)
sns.pairplot(df)
plt.show()
sns.pairplot(df[['CRIM','RM','PTRATIO','LSTAT','MEDV']])
correlation = df.corr().round(2)
correlation
plt.figure(figsize=(12,10))
sns.heatmap(data=correlation, annot=True, cmap="Blues", annot_kws={"size": 13})
plt.tight_layout()
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.scatter(df['LSTAT'],df['MEDV'])
plt.xlabel('LSTAT')
plt.subplot(1,3,2)
plt.scatter(df['RM'],df['MEDV'])
plt.xlabel('RM')
plt.subplot(1,3,3)
plt.scatter(df['PTRATIO'],df['MEDV'])
plt.xlabel('PTRATIO')

plt.show()
plt.figure(figsize=(20,5))
sns.pairplot(df, x_vars=['RM','LSTAT'], y_vars='MEDV', size=6, aspect=0.7, kind='reg')
plt.show()
df[df.RM<4.5].describe()
plt.figure(figsize=(10,8))
plt.scatter(df.CRIM, df.MEDV)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("MEDV")
hcrim = df[df.CRIM.between(*df.CRIM.quantile([0.9,1]).tolist())]
hcrim.describe()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# Split-out test dataset
array = df.values
X = array[:,0:13]
y = array[:,13]
test_size = 0.20
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# Evaluate Algorithms
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring))
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)    
    r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2') 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)  rmse= %r (%r)  r2= %a" % (name, cv_results.mean(), cv_results.std(), round(rmse.mean(),3), round(rmse.std(),3), round(r2.mean(),3))
    print(msg)
plt.figure(figsize=(12,10))
sns.boxplot(y=results,x=names).set_title('Algorithm Comparison')
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring))
    r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2') 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)  rmse= %r (%r)  r2= %a" % (name, cv_results.mean(), cv_results.std(), round(rmse.mean(),3), round(rmse.std(),3), round(r2.mean(),3))
    print(msg)
    
plt.figure(figsize=(14,10))
sns.boxplot(y=results,x=names).set_title('Scaled Algorithm Comparison')
plt.show()
# KNN Algorithm tuning

import numpy as np

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s rmse= %u" % (grid_result.best_score_, grid_result.best_params_, np.sqrt(-grid_result.best_score_)))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f)  with: %r" % (mean, stdev, param))
plt.figure(figsize=(20,8))
plt.plot(k_values,means)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Value')
plt.xticks(np.arange(0,22, step=1))
plt.show()
# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring))
    r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2') 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)  rmse= %r (%r)  r2= %a" % (name, cv_results.mean(), cv_results.std(), round(rmse.mean(),3), round(rmse.std(),3), round(r2.mean(),3))
    print(msg)
# Compare Algorithm
plt.figure(figsize=(12,10))
sns.boxplot(y=results,x=names).set_title('Scaled Ensemble Algorithm Comparison')
plt.show()
# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Tune scaled ET
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = ExtraTreesRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Make predictions on validation dataset

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
#model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model = ExtraTreesRegressor(random_state=seed, n_estimators=200)
#model = KNeighborsRegressor(n_neighbors=3)
#model = LinearRegression()
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
print("MSE : {}".format(round(mean_squared_error(y_test, predictions), 3)))
print("RMSE : {}".format(round(np.sqrt(mean_squared_error(y_test, predictions)), 3)))
print("R squared error : {}".format(round(r2_score(y_test,predictions), 3)))
sns.distplot(y_test-predictions,bins=50)

