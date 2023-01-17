# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")

data.head()
without_categorical=pd.get_dummies(data)
without_categorical.columns

del without_categorical["sex_male"]

del without_categorical["smoker_no"]

y=without_categorical["charges"]

x=without_categorical.loc[:, without_categorical.columns != 'charges']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=425)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scale=scaler.fit_transform(x_train)

x_test_scale=scaler.transform(x_test)
from sklearn.tree import DecisionTreeRegressor

dtreg=DecisionTreeRegressor()

dtreg.fit(x_train_scale,y_train)

ypred=dtreg.predict(x_test_scale)



import sklearn.metrics as metrik

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(x_train_scale,y_train)

ypred=knn.predict(x_test_scale)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)

import xgboost as xgb

xgreg=xgb.XGBRegressor()

xgreg.fit(x_train_scale,y_train)

ypred=xgreg.predict(x_test_scale)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
xgreg.get_params()
parameters={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.85 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10,],

 "min_child_weight" : [ 1, 3, 5, 7,9,11 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,0.5,0.6,0.7 ],

 "n_estimators"     : [25,50,100,150,200],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
from sklearn.model_selection import RandomizedSearchCV

clf = RandomizedSearchCV(xgb.XGBRegressor(), parameters, random_state=0)

search = clf.fit(x_train_scale,y_train)

search.best_params_
xgreg=xgb.XGBRegressor(n_estimators=100,min_child_weight=3,max_depth=3,learning_rate=0.15,gamma=0.6

                      ,colsample_bytree=0.7

                      )

xgreg.fit(x_train_scale,y_train)

ypred=xgreg.predict(x_test_scale)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
parameters2={"learning_rate"    : [0.08,0.30,0.60,0.85] ,

 "max_depth"        : [ 3,5, 6,8],

 "min_child_weight" : [ 1, 3, 5, 7],

 "gamma"            : [0.1,0.3,0.5,0.7 ],

 "n_estimators"     : [50,100,150],

 "colsample_bytree" : [ 0.3,0.5 , 0.7 ] }
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(xgb.XGBRegressor(), parameters2,n_jobs=6)

search = clf.fit(x_train_scale,y_train)

search.best_params_
xgreg=xgb.XGBRegressor(n_estimators=100,min_child_weight=7,max_depth=3,learning_rate=0.08,gamma=0.1

                      ,colsample_bytree=0.7

                      )

xgreg.fit(x_train_scale,y_train)

ypred=xgreg.predict(x_test_scale)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
xgreg.save_model("gridsearchparameters2.model")
#from tpot import TPOTRegressor

#tpot = TPOTRegressor(verbosity=2, random_state=19,max_time_mins=70)

#tpot.fit(x_train_scale, y_train)

#ypred=tpot.predict(x_test_scale)

#metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)  #2490
#tpot.export("pipeline.py")
xgreg.feature_names =data.columns



xgb.plot_importance(xgreg)
xgreg=xgb.XGBRegressor(n_estimators=100,min_child_weight=7,max_depth=3,learning_rate=0.08,gamma=0.1

                      ,colsample_bytree=0.7

                      )

xgreg.fit(x_train,y_train)

ypred=xgreg.predict(x_test)

print(metrik.mean_absolute_error(y_pred=ypred,y_true=y_test))



xgb.plot_importance(xgreg)
xgreg.save_model("betterexplained.model")
from pdpbox import pdp

pdep=pdp.pdp_isolate(xgreg,x_train,x_train.columns,"age")

pdp.pdp_plot(pdep,feature_name="age")
pdep=pdp.pdp_isolate(xgreg,x_train,x_train.columns,"bmi")

pdp.pdp_plot(pdep,feature_name="bmi")
pdep=pdp.pdp_isolate(xgreg,x_train,x_train.columns,"smoker_yes")

pdp.pdp_plot(pdep,feature_name="smoker_yes")
features=["age","bmi"]

pdp.pdp_interact_plot(xgreg,features)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg, num_trees=0,ax=ax)
fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg, num_trees=5,ax=ax)
fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg, num_trees=25,ax=ax)
fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg, num_trees=45,ax=ax)
xgreg2=xgb.XGBRegressor(n_estimators=100,min_child_weight=7,max_depth=2,learning_rate=0.08,gamma=0.1

                      ,colsample_bytree=0.7

                      )

xgreg2.fit(x_train,y_train)

ypred=xgreg2.predict(x_test)

print(metrik.mean_absolute_error(y_pred=ypred,y_true=y_test))

xgb.plot_importance(xgreg2)

xgreg2.save_model("maxdepth2.model")
pdep=pdp.pdp_isolate(xgreg2,x_train,x_train.columns,"age")

pdp.pdp_plot(pdep,feature_name="age")
pdep=pdp.pdp_isolate(xgreg2,x_train,x_train.columns,"bmi")

pdp.pdp_plot(pdep,feature_name="bmi")
fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg2, num_trees=0,ax=ax)
fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg2, num_trees=5,ax=ax)
fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_tree(xgreg2, num_trees=10,ax=ax)
import shap
shap.initjs()

explainer=shap.TreeExplainer(xgreg)
shap_values = explainer.shap_values(x_train)



shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[5,:], x_train.iloc[5,:])
shap.force_plot(explainer.expected_value, shap_values[25,:], x_train.iloc[25,:])
shap.force_plot(explainer.expected_value, shap_values[30,:], x_train.iloc[30,:])
shap.force_plot(explainer.expected_value, shap_values[40,:], x_train.iloc[40,:])
shap.summary_plot(shap_values=shap_values,feature_names=x_train.columns,plot_type="bar")
shap.summary_plot(shap_values=shap_values,alpha=0.5,feature_names=x_train.columns)
shap.dependence_plot(0, shap_values, x_train,feature_names=x_train.columns)
shap.dependence_plot(1, shap_values, x_train,feature_names=x_train.columns)
shap.dependence_plot(2, shap_values, x_train,feature_names=x_train.columns)
shap.dependence_plot(3, shap_values, x_train,feature_names=x_train.columns)
explainer=shap.TreeExplainer(xgreg2)

shap_values = explainer.shap_values(x_train)



shap.force_plot(explainer.expected_value, shap_values[0,:], x_train.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[15,:], x_train.iloc[15,:])
shap.force_plot(explainer.expected_value, shap_values[25,:], x_train.iloc[25,:])
shap.summary_plot(shap_values=shap_values,feature_names=x_train.columns,plot_type="bar")
shap.summary_plot(shap_values=shap_values,alpha=0.5,feature_names=x_train.columns)
shap.dependence_plot(1, shap_values, x_train,feature_names=x_train.columns)
shap.dependence_plot(0, shap_values, x_train,feature_names=x_train.columns)
shap.dependence_plot(0, shap_values, x_train,feature_names=x_train.columns)