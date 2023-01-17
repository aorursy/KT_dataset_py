import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
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
import pickle
print(os.listdir("../input"))

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv("../input/housing.csv",delim_whitespace=True,names=names)
print(dataset.shape)
print(dataset.dtypes)
#pd.set_option('display.width',500)
print(dataset.head(5))
dataset.describe()
dataset.corr()
dataset.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
plt.show()
dataset.plot(kind="density",subplots=True,layout=(4,4),sharex=False,sharey=False,fontsize=1)
plt.show()
dataset.plot(kind="box",subplots=True,layout=(4,4),sharex=False,sharey=False,fontsize=1)
plt.show()
pd.plotting.scatter_matrix(dataset,figsize=(10,10))
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(),vmin=-1,vmax=1,interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.2
seed = 7
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)
num_folds = 10
scoring = 'neg_mean_squared_error'
models = {}
models["LR"] = LinearRegression()
models["LASSO"] = Lasso()
models["EN"] = ElasticNet()
models["CART"] = DecisionTreeRegressor()
models["KNN"] = KNeighborsRegressor()
models["SVM"] = SVR()
results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print("%s: %f, %f" %(key,cv_result.mean(),cv_result.std()))
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
pipelines = {}
pipelines["ScalarLR"] = Pipeline([("Scalar",StandardScaler()),("LR",LinearRegression())])
pipelines["ScalarLASSO"] = Pipeline([("Scalar",StandardScaler()),("LASSO",Lasso())])
pipelines["ScalarEN"] = Pipeline([("Scalar",StandardScaler()),("EN",ElasticNet())])
pipelines["ScalarKNN"] = Pipeline([("Scalar",StandardScaler()),("KNN",KNeighborsRegressor())])
pipelines["ScalarCART"] = Pipeline([("Scalar",StandardScaler()),("CART",DecisionTreeRegressor())])
pipelines["ScalarSVM"] = Pipeline([("Scalar",StandardScaler()),("SVM",SVR())])

results = []
for key in pipelines:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(pipelines[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print("%s: %f, %f" %(key,cv_result.mean(),cv_result.std()))
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(pipelines.keys())
plt.show()
scaler = StandardScaler().fit(X_train)
rescaleX = scaler.transform(X_train)
param_grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescaleX,y=Y_train)
print("best_score:",grid_result.best_score_)
print("best_params:",grid_result.best_params_)
ensembers = {}
ensembers["ScalarAB"] = Pipeline([("Scalar",StandardScaler()),("AB",AdaBoostRegressor())])
ensembers["ScalarABKNN"] = Pipeline([("Scalar",StandardScaler()),("ABKNN",AdaBoostRegressor(KNeighborsRegressor(n_neighbors=3)))])
ensembers["ScalarABLR"] = Pipeline([("Scalar",StandardScaler()),("ABLR",AdaBoostRegressor(LinearRegression()))])
ensembers["ScalarRFT"] = Pipeline([("Scalar",StandardScaler()),("RFT",RandomForestRegressor())])
ensembers["ScalarETR"] = Pipeline([("Scalar",StandardScaler()),("ETR",ExtraTreesRegressor())])
ensembers["ScalarGRB"] = Pipeline([("Scalar",StandardScaler()),("GRB",GradientBoostingRegressor())])
results = []
for key in ensembers:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_result = cross_val_score(ensembers[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print("%s: %f, %f" %(key,cv_result.mean(),cv_result.std()))
fig = plt.figure(figsize=(10,10))
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembers.keys())
plt.show()
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]}
model = GradientBoostingRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80]}
model = ExtraTreesRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=Y_train)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
#训练模型
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
gbr = ExtraTreesRegressor(n_estimators=80)
gbr.fit(X=rescaledX, y=Y_train)
# 评估算法模型
rescaledX_validation = scaler.transform(X_validation)
predictions = gbr.predict(rescaledX_validation)
print(mean_squared_error(Y_validation, predictions))
model_file = "model.sav"
with open(model_file,mode='wb') as model_f:
    pickle.dump(gbr,model_f)
