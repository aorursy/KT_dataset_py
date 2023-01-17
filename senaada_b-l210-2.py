import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from argparse import Namespace
import seaborn as sns 
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
startup = pd.read_csv("../input/50startups/50_Startups.csv", sep = ",")
df=startup.copy()
df.head()
df.info()
df.shape
df.isna().sum()
df.corr()
corr=df.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values);
sns.scatterplot(x="R&D Spend" , y="Profit" , data=df);
df.hist()
df.describe().T
df["State"].unique()
df_State=pd.get_dummies(df["State"])
df_State.head()
df_State.columns = ['California', 'Florida', 'New York']
df_State.head()
df = pd.concat([df, df_State], axis = 1)
df
df.head()
X=df.drop(["Florida" , "State"] , axis=1)
y=df["Profit"]
X
(X,y)
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
X_test
y_train
y_test
linear_regressor = LinearRegression()
linear_regressor.fit(X_train , y_train)
y_pred = linear_regressor.predict(X_test)
y_pred
df["tahmin"]=linear_regressor.predict(X)
df
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error
MAE
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error
MSE
linear_regressor.score(X,y)