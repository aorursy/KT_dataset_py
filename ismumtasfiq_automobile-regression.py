import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data ",names=['symboling','normalize_losses','make','fueal_type','aspiration','num_of_doors','body_style','drive_wheels','engine_locatio','wheel_base'
     ,'length','width','height','curb_weight','engine_type','num_of_cylinders','engine_size','fuel_system'
     ,'bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price'],na_values="?")
data.head()
data.dtypes
data.isnull().any()
data[data.isnull().any(axis=1)][data.columns[data.isnull().any()]]
data.dtypes
data.drop(["symboling","normalize_losses","make","fueal_type","fuel_system","aspiration","num_of_doors","body_style","drive_wheels","engine_locatio","engine_type"],axis=1,inplace=True)
data
data.dtypes
data["num_of_cylinders"].unique()
data["cylinders"]=data["num_of_cylinders"].replace({'four':4, 'six':6, 'five':5, 'three':3, 'twelve':12, 'two':2, 'eight':8})

data["cylinders"]
data.drop(['num_of_cylinders'],axis=1,inplace=True)
data.head()
data.describe()
data.isnull().any()
data["bore"]=data["bore"].fillna(data["bore"].median())
data["stroke"]=data["stroke"].fillna(data["stroke"].median())
data["horsepower"]=data["horsepower"].fillna(data["horsepower"].median())
data["peak_rpm"]=data["peak_rpm"].fillna(data["peak_rpm"].median())
data["price"]=data["price"].fillna(data["price"].median())
data.isnull().any()
sns.pairplot(data)
sns.distplot(data["price"])
X=data.drop("price",axis=1)
y=data[["price"]]
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=1)
regression_model=LinearRegression()
regression_model.fit(X_train,y_train)
X_train.columns
for i,col_name in enumerate(X_train.columns):
    print(" coefficient of {} is {}".format(col_name,regression_model.coef_[0][i]))
intercept=regression_model.intercept_[0]
print(" intercept for model ",intercept)
regression_model.score(X_test,y_test)

import statsmodels.formula.api as smf
cars=pd.concat([y_train,X_train],axis=1)
cars
incars=smf.ols(formula='price~wheel_base + length + width+ height+ curb_weight+ engine_size+bore+stroke+ compression_ratio+ horsepower +peak_rpm+ city_mpg+ highway_mpg+ cylinders',data=cars).fit()






incars.params
print(incars.summary())
