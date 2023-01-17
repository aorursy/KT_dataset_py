import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head(10)
df.isnull().sum()
df.drop(['name','host_name'],axis = 1, inplace = True)
df.dropna(axis = 0,inplace = True)
new_df = pd.get_dummies(df,columns = ['neighbourhood_group','neighbourhood','room_type'])
new_df
X = new_df.drop(['price'], axis = 1)
y = new_df.price
X = new_df.drop(['id','host_id','last_review'],axis =1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 43)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models =[LinearRegression(),SVR(),DecisionTreeRegressor(),RandomForestRegressor(),GradientBoostingRegressor()]
a,b,c,d,e = [],[],[],[],[]
for i in models:
    model = i.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    a.append(mean_squared_error(y_test,y_pred))
    b.append(mean_absolute_error(y_test,y_pred))
    c.append(r2_score(y_test,y_pred))
    d.append(math.sqrt(mean_squared_error(y_test,y_pred)))
    e.append(1 - (1-r2_score(y_test,y_pred)) * (len(X_test)-1)/(len(X_test)-len(X.columns)-1))

pd.DataFrame([a,b,c,d,e],index = ['MSE','MAE','R2','RMSE','Adjusted R2'], columns = ['Linear Reg','SVR','Decision Tree Reg','Random Forest Reg','Gradient Boosting Regressor'])
