import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
with open("../input/housing.csv","r") as f:

    data=f.readlines()



housing_data=[]

for line in data:

    samples=[np.float32(x) for x in line.split()]

    housing_data.append(samples)



housing_data=np.asarray(housing_data)

boston=pd.DataFrame(housing_data,columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LTSTAT","MEDV"])

print(boston.head())
from pandas.plotting import scatter_matrix



scatter_matrix(boston,figsize=(16,16))

plt.show()
import seaborn as sns

cor=boston.corr()

fig=plt.figure(figsize=(12,12))

fig=sns.heatmap(cor,annot=True)

plt.show()

X1=boston.loc[:,["RM","PTRATIO","LTSTAT","INDUS","NOX","TAX"]]



y=boston["MEDV"]
print("X\n",X1.head(),"\n\nY\n",y.head())
print(X1.describe())
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Lasso

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.tree import DecisionTreeRegressor



from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV

seed=35

kfold=KFold(n_splits=10,random_state=seed)

scoring="r2"
x_train,x_test,y_train,y_test=train_test_split(X1,y,test_size=.3,random_state=seed)
models=[]

models.append(["LR",LinearRegression()])

models.append(["ENet",ElasticNet()])



models.append(["SVR",SVR(gamma="scale")])

models.append(["KNR",KNeighborsRegressor()])

models.append(["GPR",GaussianProcessRegressor(normalize_y=True)])

models.append(["CART",DecisionTreeRegressor()])
param_grids=[]

LR_param_grid={}

param_grids.append(LR_param_grid)

ENet_param_grid={}

ENet_param_grid["alpha"]=[.001,.01,.1,.3,.5]

ENet_param_grid["l1_ratio"]=[0,.2,.4,.5,.7,1]

param_grids.append(ENet_param_grid)

svr_param_grid={}

svr_param_grid["kernel"]=["poly","linear","rbf"]

svr_param_grid["degree"]=[1,2,3,4]

svr_param_grid["C"]=[.001,0.1,.3,.5,1,2,3]

param_grids.append(svr_param_grid)

knr_param_grid={}

knr_param_grid["n_neighbors"]=[3,5,7,11]

knr_param_grid["weights"]=["uniform","distance"]

param_grids.append(knr_param_grid)

gpr_param_grid={}

param_grids.append(gpr_param_grid)

cart_param_grid={}

cart_param_grid["max_depth"]=[1,2,3,4]

param_grids.append(cart_param_grid)
results=[['LR', {}, 0.6743397010174477], ['ENet', {'alpha': 0.1, 'l1_ratio': 0.7}, 0.6758633980611956], ['SVR', {'C': 0.3, 'degree': 1, 'kernel': 'linear'}, 0.6646899021487152], ['KNR', {'n_neighbors': 3, 'weights': 'distance'}, 0.7612886361653691], ['GPR', {}, -13.56991654534516], ['CART', {'max_depth': 3}, 0.7346860566317643]]
print(results)
test_scores=[]

for model,result in zip(models,results):

    clf=model[1]

    clf.set_params(**result[1])

    clf.fit(x_train,y_train)

    score=clf.score(x_test,y_test)

    test_scores.append([model[0],result[1],score])



for model,param,score in test_scores:

    print("%s : %0.4f"%(model,score))
from sklearn.ensemble import BaggingRegressor

bgr_param_dict={"n_estimators":list(np.arange(1,100,5)),"max_samples":list(np.linspace(.1,1,10)),"random_state":[seed]}

knn=KNeighborsRegressor(n_neighbors=3,weights="distance")

bg=BaggingRegressor(base_estimator=knn)
bagging_best_params = {'max_samples': 1.0, 'n_estimators': 76, 'random_state': 35}

bagging_best_score=0.764816
print("best training score : %0.6f\nbest params : %r"%(bagging_best_score,bagging_best_params))

bg.set_params(**bagging_best_params)

bg.fit(x_train,y_train)

print("Bagging Test score : %0.6f"%bg.score(x_test,y_test))
knn.fit(x_train,y_train)

print(knn.score(x_test,y_test))
from sklearn.ensemble import RandomForestRegressor

params_dict={}

params_dict["n_estimators"]=list(np.arange(1,100,5))

params_dict["max_depth"]=[None,2,3,4,5,6,7,8,9,10]

params_dict["max_features"]=[.2,.6,.8,1.0]

params_dict["bootstrap"]=[True]

params_dict["random_state"]=[seed]

#print(gcv.best_params_)

rfc_best_params={'bootstrap': True,

 'max_depth': 9,

 'max_features': 0.2,

 'n_estimators': 96,

 'random_state': 35}

print(rfc_best_params)
#print(gcv.best_score_)

rfc_best_score=0.8607639890718424

print(rfc_best_score)
rfc=RandomForestRegressor()

rfc.set_params(**rfc_best_params)

rfc.fit(x_train,y_train)
from sklearn.ensemble import ExtraTreesRegressor
#print(gcv.best_params_)

et_best_params={'bootstrap': True,

 'max_depth': None,

 'max_features': 0.6,

 'n_estimators': 71,

 'random_state': 35}

print(et_best_params)
#print(gcv.best_score_)

et_best_score=0.870845214813861

print(et_best_score)
et=ExtraTreesRegressor()

et.set_params(**et_best_params)

et.fit(x_train,y_train)
print(et.score(x_test,y_test))
evaluated_models=['LR', 'ENet', 'SVR', 'KNR', 'GPR', 'CART', 'BAGGING', 'RForest','ETrees']

evaluated_test_scores= [0.6077, 0.6012,0.587,0.6843,-0.0222,0.6223,0.6673,0.8172,0.8490]   
plt.figure(figsize=(8,6))

plt.plot(evaluated_models,evaluated_test_scores,marker="o",linestyle="--",color="r")

plt.ylim(-0.5,1)

plt.title("Model Evaluation Plot")

plt.xlabel("Models")

plt.ylabel("R-Score")

plt.show()