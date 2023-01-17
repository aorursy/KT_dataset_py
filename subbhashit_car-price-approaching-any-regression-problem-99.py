import numpy as np
import pandas as pd
data=pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv")
data.head()
data.describe()
import seaborn as sns
sns.pairplot(data)
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.distplot(data['price'],color="y")
sns.jointplot(x="wheelbase",y="price",data=data,kind='kde',color='red')
plt.figure(figsize=(15,10))
sns.countplot(data['enginelocation'])
plt.figure(figsize=(15,10))
sns.scatterplot(data.enginelocation,data['price'],color=['c'])
data.info()
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['fuelsystem']=lab.fit_transform(data['fuelsystem'])
data['cylindernumber']=lab.fit_transform(data['cylindernumber'])
data['enginetype']=lab.fit_transform(data['enginetype'])
data['enginelocation']=lab.fit_transform(data['enginelocation'])
data['drivewheel']=lab.fit_transform(data['drivewheel'])
data['carbody']=lab.fit_transform(data['carbody'])
data['doornumber']=lab.fit_transform(data['doornumber'])
data['aspiration']=lab.fit_transform(data['aspiration'])
data['fueltype']=lab.fit_transform(data['fueltype'])
data['CarName']=lab.fit_transform(data['CarName'])
data.head()
import missingno as msno
msno.dendrogram(data)
sns.boxplot(data['enginesize'],color='red')
data['enginesize']=data['enginesize'].rank()
sns.boxplot(data['enginesize'],color='red')
corr=data.corr()
corr.style.background_gradient(cmap="inferno")
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,cmap='inferno',mask=np.triu(data.corr(),k=1))
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
x=data.drop(['car_ID','price'],axis=1)
y=data['price']
xr,xt,yr,yt=train_test_split(x,y,test_size=0.1)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error,make_scorer
from sklearn.pipeline import make_pipeline
mod=LGBMRegressor(n_estimators=40)
model=make_pipeline(mod)
model.fit(x,y)
print(model)
kfold=KFold(n_splits=5)
score=cross_val_score(model,x,y,cv=kfold)
print(score)
yp=model.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))
mod=RandomForestRegressor(n_estimators=100)
model=make_pipeline(mod)
print(model)
kfold=KFold(n_splits=5)
model.fit(x,y)
score=cross_val_score(model,x,y,cv=kfold)
print(score)
yp=model.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))
xgb1 = XGBRegressor()
parameters = {'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,parameters,cv = 2)

xgb_grid.fit(x,y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
yp=xgb_grid.predict(xt)
print(r2_score(yt,yp))
print(mean_squared_error(yt,yp))
print(mean_squared_log_error(yt,yp))