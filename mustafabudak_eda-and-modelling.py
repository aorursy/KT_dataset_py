# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,GradientBoostingRegressor
data=pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv")
df=data.copy()
df.head()
df.info()
df.drop(["Unnamed: 0"],1,inplace=True)
df=df.rename(columns={"parking spaces":"parking_spaces",
             "rent amount":"rent_amount",
             "property tax":"property_tax",
             "fire insurance":"fire_insurance"})

df.head()
def removing(x):
    s =  x[2:] #removes first two chr
    snc = ""
    for i in s:
        if i.isdigit() is True:
            snc = snc + i
    return snc

df["hoa"] = pd.to_numeric(df["hoa"].apply(removing), errors= "ignore")
df["rent_amount"] = pd.to_numeric(df["rent_amount"].apply(removing), errors= "ignore")
df["property_tax"] = pd.to_numeric(df["property_tax"].apply(removing), errors= "ignore")
df["fire_insurance"] = pd.to_numeric(df["fire_insurance"].apply(removing), errors= "ignore")
df["total"] = pd.to_numeric(df["total"].apply(removing), errors= "ignore")
df.head()
df.describe()
df["floor"]=df['floor'].replace('-', np.nan)

df["floor"].fillna(df["floor"].median(),inplace=True)
df["hoa"].fillna(df["hoa"].mean(),inplace=True)
df["property_tax"].fillna(df["property_tax"].mean(),inplace=True)

df["floor"]=df["floor"].astype("int64")
df["furniture"]=df["furniture"].astype("category")

df["furniture"]=[1 if i=="furnished" else 0 for i in df["furniture"]]
df.drop(["animal"],1,inplace=True)

df.isnull().sum()
df.dtypes
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
plt.show()
sns.relplot(x="fire_insurance", y="rent_amount", hue="bathroom",
            data=df);
plt.title("Fire Insurance-Rent Amount with Bathroom",color="red")
plt.xlabel("Fire Insurance")
plt.ylabel("Rent Amount")
plt.show()
g = sns.FacetGrid(df, hue="rooms",size=7)
g.map(plt.scatter, "hoa", "total", alpha=.7)
g.add_legend();
plt.title("Hoa-Total Relationship")
g = sns.FacetGrid(df,hue="furniture",palette="Set1", height=5, hue_kws={"marker": ["^", "v"]},size=7)
g.map(plt.scatter, "fire_insurance", "total", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
plt.figure(figsize=(12,6))
sns.pointplot(x="bathroom",y="total",hue="parking_spaces",data=df)
plt.title("Bathroom-Total Relationship with Parking Spaces")
g = sns.FacetGrid(df,col="furniture", hue="floor",size=7)
g.map(plt.scatter, "rent_amount", "total", alpha=.7)
g.add_legend();
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x=df['rooms'], y=df['total'],color='red',alpha=0.8)
sns.pointplot(x=df['floor'], y=df['total'],color='lime',alpha=0.8)
plt.xlabel('Rooms-Floor',fontsize = 15,color='blue')
plt.ylabel('Total',fontsize = 15,color='blue')
plt.title('Floor and Rooms relationship with Total',fontsize = 20,color='blue')
plt.grid()
df.drop(["furniture","city","area","floor"],1,inplace=True)
df.drop(["fire_insurance"],1,inplace=True)
total_df=df["total"].copy()

q1=total_df.quantile(0.25)
q3=total_df.quantile(0.75)
IQR=q3-q1

l_bound=q1-1.5*IQR
u_bound=q3+1.5*IQR

print(l_bound)
print(u_bound)

table_min=total_df.min()
table_max=total_df.max()

for e in range(len(total_df)):
    if total_df.iloc[e]<l_bound:
        total_df.iloc[e]=l_bound
        
    elif total_df.iloc[e] >u_bound:
        total_df.iloc[e]=u_bound
        

df["total"]=total_df
# lets check it
sns.boxplot(df["total"])
X=df.drop(["total"],1)
y=df["total"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
lm=sm.OLS(y,X)
model=lm.fit()
model.summary()
lr_model=LinearRegression()
lr_model.fit(X_train,y_train)
lr_pred=lr_model.predict(X_test)

print("Test error:",np.sqrt(mean_squared_error(y_test,lr_pred)))
print("Train error:",np.sqrt(mean_squared_error(y_train,lr_model.predict(X_train))))
print("score:",lr_model.score(X_train,y_train))
knn_params={"n_neighbors":np.arange(1,30,1)}
knn=KNeighborsRegressor()
knn_cv=GridSearchCV(knn,knn_params,cv=10)
knn_cv.fit(X_train,y_train)
knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv.best_params_["n_neighbors"]).fit(X_train,y_train)
print("test error:",np.sqrt(mean_squared_error(y_test,knn_tuned.predict(X_test))))
print("score:",knn_tuned.score(X_train,y_train))
cart=DecisionTreeRegressor()
cart.fit(X_train,y_train)
cart_params={"min_samples_split":range(2,100),
            "max_leaf_nodes":range(2,10)}
cart_cv=GridSearchCV(cart,cart_params,cv=10,n_jobs=-1,verbose=2)
cart_cv.fit(X_train,y_train)
cart_tuned=DecisionTreeRegressor(min_samples_split=cart_cv.best_params_["min_samples_split"],
                                 max_leaf_nodes=cart_cv.best_params_["max_leaf_nodes"]).fit(X_train,y_train)
print("test error:",np.sqrt(mean_squared_error(y_test,cart_tuned.predict(X_test))))
print("score:",cart_tuned.score(X_test,y_test))
Importance = pd.DataFrame({"Importance": cart_tuned.feature_importances_*100},
                         index = X_train.columns)

Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable importance types")
plt.title("Variable importance values")
bag_model=BaggingRegressor(bootstrap_features=True).fit(X_train,y_train)
bag_params={"n_estimators":range(2,20)}
bag_cv=GridSearchCV(bag_model,bag_params,cv=10,verbose=2,n_jobs=-1)
bag_cv.fit(X_train,y_train)
bag_tuned=BaggingRegressor(bootstrap_features=True,n_estimators=bag_cv.best_params_["n_estimators"]).fit(X_train,y_train)
print("test error:",np.sqrt(mean_squared_error(y_test,bag_tuned.predict(X_test))))
print("score:",bag_tuned.score(X_test,y_test))
rf_model=RandomForestRegressor().fit(X_train,y_train)
rf_params={"max_depth":list(range(2,9)),
          "max_features":[3,5,10,15],
          "n_estimators":[100,200,500,1000]}

rf_cv=GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2)
rf_cv.fit(X_train,y_train)
rf_tuned=RandomForestRegressor(max_depth=rf_cv.best_params_["max_depth"],
                               max_features=rf_cv.best_params_["max_features"],
                               n_estimators=rf_cv.best_params_["n_estimators"]).fit(X_train,y_train)
print("test error:",np.sqrt(mean_squared_error(y_test,rf_tuned.predict(X_test))))
print("score:",rf_tuned.score(X_train,y_train))
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)

Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable importance types")
from xgboost import XGBRegressor

xgb=XGBRegressor().fit(X_train,y_train)
xgb_params={
     'n_estimators':[100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]}

xgb_cv=GridSearchCV(xgb,xgb_params,cv=10,verbose=2,n_jobs=-1).fit(X_train,y_train)  
xgb_tuned=XGBRegressor(
                       n_estimators=xgb_cv.best_params_["n_estimators"],
                       max_depth=xgb_cv.best_params_["max_depth"],
                       learning_rate=xgb_cv.best_params_["learning_rate"] )
xgb_tuned.fit(X_train,y_train)
print("test error:",np.sqrt(mean_squared_error(y_test,xgb_tuned.predict(X_test))))
print("score:",xgb_tuned.score(X_test,y_test))                          
from lightgbm import LGBMRegressor

lgbm_grid = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.5,1],
    'n_estimators': [20, 40, 100, 200, 500,1000],
    'max_depth': [1,2,3,4,5,6,7,8] }

lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs = -1, verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = lgbm_cv_model.best_params_["learning_rate"], 
                           max_depth = lgbm_cv_model.best_params_["max_depth"], 
                           n_estimators = lgbm_cv_model.best_params_["n_estimators"],
                          colsample_bytree = lgbm_cv_model.best_params_["colsample_bytree"])

lgbm_tuned = lgbm_tuned.fit(X_train,y_train)
print("test error:",np.sqrt(mean_squared_error(y_test, lgbm_tuned.predict(X_test))))
print("score:",lgbm_tuned.score(X_test,y_test))

