import numpy as np 
import pandas as pd 
from collections import defaultdict,OrderedDict
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")


import os
print(os.listdir("../input"))
data = pd.read_csv("../input/data.csv")
data.info()
data = data.drop(axis=0,columns="Rank")
data["Subscribers"] = data["Subscribers"].apply(lambda x:x.replace("-- ","0")).astype(int)
data["Video Uploads"] = data["Video Uploads"].apply(lambda x:x.replace("--","0")).astype(int)
data.info()
data.head()
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax)
plt.show()
plt.figure(figsize=(12,8))
sns.regplot(x=data["Subscribers"], y=data["Video views"], color = 'purple')
plt.show()
X = data[["Video Uploads","Video views"]]
y = data[["Subscribers"]]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

lr = LinearRegression()
lr.fit(X,y)
y_pred = lr.predict(X)

score = r2_score(y,y_pred)
mse = mean_squared_error(y,y_pred)
print("r2 score :{} mse : {}".format(score,mse))
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pf = PolynomialFeatures(degree = 2)
X_poly = pf.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly,y)
y_pred = lr.predict(X_poly)

score = r2_score(y,y_pred)
mse = mean_squared_error(y,y_pred)
print("r2 score :{} mse : {}".format(score,mse))
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X = ss.fit_transform(X)
y= ss.fit_transform(y)

svr_regressor = SVR(kernel = "rbf")
svr_regressor.fit(X,y)
y_pred = svr_regressor.predict(X)

score = r2_score(y,y_pred)
mse = mean_squared_error(y,y_pred)
print("r2 score :{} mse : {}".format(score,mse))
from sklearn.tree import DecisionTreeRegressor


dt = DecisionTreeRegressor(random_state = 42,max_depth=10)
dt.fit(X,y)
y_pred = dt.predict(X)

score = r2_score(y,y_pred)
mse = mean_squared_error(y,y_pred)
print("r2 score :{} mse : {}".format(score,mse))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42, n_estimators = 7, max_depth = 20)
rf.fit(X,y)
y_pred = rf.predict(X)

score = r2_score(y,y_pred)
mse = mean_squared_error(y,y_pred)
print("r2 score :{} mse : {}".format(score,mse))