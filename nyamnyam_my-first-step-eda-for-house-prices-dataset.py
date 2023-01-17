import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

dat = pd.read_csv("../input/train.csv")
dat = dat.drop("Id",axis=1) #Id is dropped

print("number of observations:{}".format(dat.shape[0]))
print("number of variables:{}".format(dat.shape[1]))
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,7.5))
ax1=sns.distplot(dat.SalePrice,ax=ax1)
ax1.set_title("raw data")
ax2=sns.distplot(np.log(dat.SalePrice),ax=ax2)
ax2.set_title("log-transformed")
corr_mat = dat.corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_mat, vmax=.8, square=True)
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
#fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,figsize=(12,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
ax1.scatter(dat.OverallQual,dat.SalePrice)
ax1.set_title("OverallQual")
ax2.scatter(dat.YearBuilt,dat.SalePrice)
ax2.set_title("YearBuilt")
ax3.scatter(dat.FullBath,dat.SalePrice)
ax3.set_title("FullBath")
ax4.scatter(dat.TotRmsAbvGrd,dat.SalePrice)
ax4.set_title("TotRmsAbvGrd")
ax5.scatter(dat.GarageCars,dat.SalePrice)
ax5.set_title("GarageCars")
ax6.scatter(dat.GarageArea,dat.SalePrice)
ax6.set_title("GarageArea")
dat_sub = pd.concat([dat['SalePrice'], dat["Neighborhood"]], axis=1)
f, ax = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
sns.boxplot(x="Neighborhood", y="SalePrice", data=dat_sub)
plt.xticks(rotation=90)

dat_sub = pd.concat([dat['SalePrice'], dat["Street"]], axis=1)
f, ax = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
sns.boxplot(x="Street", y="SalePrice", data=dat_sub)
plt.xticks(rotation=90)

dat_sub = pd.concat([dat['SalePrice'], dat["HouseStyle"]], axis=1)
f, ax = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
sns.boxplot(x="HouseStyle", y="SalePrice", data=dat_sub)
plt.xticks(rotation=90)



total_nulls = dat.isnull().sum().sort_values(ascending=False)
percent_nulls = (dat.isnull().sum()/dat.isnull().count()).sort_values(ascending=False)
dat_missing = pd.concat([total_nulls, percent_nulls], axis=1, keys=['Total', 'Percent'])
dat_missing[dat_missing["Percent"]>0]
dat_comp = dat.drop((dat_missing[dat_missing["Percent"] > 0]).index,1)
dat_comp = pd.get_dummies(dat_comp)
dat_comp.head()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(dat_comp.drop("SalePrice", axis=1), dat_comp.SalePrice, random_state=1234,train_size=0.9)
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

param = {
    "alpha": np.linspace(0, 1000, 100)
}

model_lasso = Lasso()
tune_lasso = GridSearchCV(model_lasso, param,n_jobs=-1)
tune_lasso.fit(train_X, train_y)
best_lasso = tune_lasso.best_params_
print(tune_lasso.best_estimator_)
print("Validation accuracy (LASSO): {}".format(tune_lasso.score(test_X,test_y)))
plt.scatter(tune_lasso.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()
from sklearn.linear_model import Ridge
param = {
    "alpha": np.linspace(0, 1000, 100)
}

model_ridge = Ridge()
tune_ridge = GridSearchCV(model_ridge, param,n_jobs=-1)
tune_ridge.fit(train_X, train_y)
best_ridge = tune_ridge.best_params_
print("Validation accuracy (LASSO): {}".format(tune_ridge.score(test_X,test_y)))
plt.scatter(tune_ridge.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()
from sklearn.ensemble import RandomForestRegressor

param = {
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [10,20,30],
    "random_state": [i for i in range(20)]
}

model_rf = RandomForestRegressor()
tune_rf = GridSearchCV(model_rf, param,n_jobs=-1)
tune_rf.fit(train_X, train_y)
best_rf = tune_rf.best_params_
print(tune_rf.best_estimator_)
print("Validation accuracy (RandomForest): {}".format(tune_rf.score(test_X,test_y)))
plt.scatter(tune_rf.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()
from sklearn.ensemble import AdaBoostRegressor

param = {
    "loss": ["linear", "square", "exponential"],
    "learning_rate": np.linspace(0.1, 1, 10),
    "n_estimators": [10,20,30]
}

model_ab = AdaBoostRegressor()
tune_ab = GridSearchCV(model_ab, param,n_jobs=-1)
tune_ab.fit(train_X, train_y)
best_ab = tune_ab.best_params_
print(tune_ab.best_estimator_)
print("Validation accuracy (Adaboost): {}".format(tune_ab.score(test_X,test_y)))
plt.scatter(tune_ab.predict(train_X),train_y)
plt.xlabel("predict")
plt.ylabel("observed")
plt.show()
feat_imp = tune_rf.best_estimator_.feature_importances_
feat_imp = pd.DataFrame({
                    "variable":train_X.columns,
                    "importance":feat_imp
})

feat_imp = feat_imp.sort_values("importance",ascending=False)
feat_imp.head(10)
dat_test = pd.read_csv("../input/test.csv")
Ids = dat_test["Id"]
dat_test = dat_test.drop("Id",axis=1) #Id is dropped

#preprocessing for test dataset
dat_test_comp = dat_test.drop((dat_missing[dat_missing["Percent"] > 0]).index,1)
dat_test_comp = pd.get_dummies(dat_test_comp)
dat_test_comp.fillna(method='ffill',inplace=True)

#add missing categories as dummy variables for test set
notintest_vals = set(dat_comp)-set(dat_test_comp)
for val in notintest_vals: 
    if val != "SalePrice":
        dat_test_comp[val] = np.zeros(dat_test_comp.shape[0])
        
#apply RF regression
pred_test = tune_rf.predict(dat_test_comp)
pred_table = pd.DataFrame({
    "Id": Ids,
    "SalePrice": pred_test
})
pred_table.to_csv("predict.csv",index=False)
