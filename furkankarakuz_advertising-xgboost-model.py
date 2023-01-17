# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/advertising/advertising.csv")

df=df.iloc[:,1:]
df.head()
df.tail()
df.info()
df.corr()
fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(16,12))

list_col=list(df.columns)

color=["#DC143C","#008000","#4169E1","#FFA500"]

col=0

for i in range(2):

    for j in range(2):

        sns.distplot(df[list_col[col]],ax=ax[i][j],kde=False,color=color[col],bins=30)

        col+=1
fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(16,12))

col=0

for i in range(2):

    for j in range(2):

        sns.kdeplot(df[list_col[col]],ax=ax[i][j],color=color[col],shade=True)

        col+=1
fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(16,5))

col=0

for i in range(3):

    sns.kdeplot(df[list_col[i]],df["sales"],ax=ax[i],color=color[i],shade=True)

    col+=1
fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(16,5))

col=0

for i in range(3):

    sns.scatterplot(data=df,x=list_col[i],y="sales",ax=ax[i],color=color[i],s=80)

    col+=1
fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(16,5))

col=0

for i in range(3):

    sns.regplot(data=df,x=list_col[i],y="sales",ax=ax[i],scatter_kws={"s":80},color=color[i],line_kws={"lw":5,"color":color[-1]})
plt.figure(figsize=(16,6))

for i in range(3):

    sns.scatterplot(data=df,x=list_col[i],y="sales",s=80,color=color[i])
plt.figure(figsize=(16,6))

for i in range(3):

    sns.regplot(data=df,x=list_col[i],y="sales",scatter_kws={"s":80},color=color[i],line_kws={"lw":5})



plt.xlim(-10,300)
sns.pairplot(df,kind="reg",aspect=1.8,height=1.8)
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from xgboost import XGBRegressor
X=df.drop(["sales"],axis=1)

Y=df["sales"]



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(Y_train.shape,Y_test.shape)
xgb_model=XGBRegressor().fit(X_train,Y_train)
r2_score(Y_train,xgb_model.predict(X_train))
mae=mean_absolute_error(Y_train,xgb_model.predict(X_train))

mse=mean_squared_error(Y_train,xgb_model.predict(X_train))

rmse=np.sqrt(mse)



print("Mean Absolute Error (MAE) : ",mae)

print("Mean Squared Error (MSE)  :",mse)

print("Mean Squared Error (MSE)  :",rmse)
mae=mean_absolute_error(Y_test,xgb_model.predict(X_test))

mse=mean_squared_error(Y_test,xgb_model.predict(X_test))

rmse=np.sqrt(mse)



print("Mean Absolute Error (MAE) : ",mae)

print("Mean Squared Error (MSE)  :",mse)

print("Mean Squared Error (MSE)  :",rmse)
Importance=pd.DataFrame({"Importance":xgb_model.feature_importances_},index=X_train.columns)

Importance.plot.barh(color=color[-1])
xgb_model
xgb_params={"colsample_bytree":[0.2,0.4,0.8,1],

           "n_estimators":[100,200,500,1000],

           "max_depth":[2,4,6],

           "learning_rate":[0.01,0.1,0.3,0.5]}
xgb_cv=GridSearchCV(xgb_model,xgb_params,cv=5,n_jobs=-1,verbose=2).fit(X_train,Y_train)
xgb_cv.best_params_
xgb_model=XGBRegressor(colsample_bytree=1,learning_rate=0.01,max_depth=4,n_estimators=1000).fit(X_train,Y_train)
r2_score(Y_train,xgb_model.predict(X_train))
mae=mean_absolute_error(Y_train,xgb_model.predict(X_train))

mse=mean_squared_error(Y_train,xgb_model.predict(X_train))

rmse=np.sqrt(mse)



print("Mean Absolute Error (MAE) : ",mae)

print("Mean Squared Error (MSE)  :",mse)

print("Mean Squared Error (MSE)  :",rmse)
mae=mean_absolute_error(Y_test,xgb_model.predict(X_test))

mse=mean_squared_error(Y_test,xgb_model.predict(X_test))

rmse=np.sqrt(mse)



print("Mean Absolute Error (MAE) : ",mae)

print("Mean Squared Error (MSE)  :",mse)

print("Mean Squared Error (MSE)  :",rmse)
Importance=pd.DataFrame({"Importance":xgb_model.feature_importances_},index=X_train.columns)

Importance.plot.barh(color=color[-1])