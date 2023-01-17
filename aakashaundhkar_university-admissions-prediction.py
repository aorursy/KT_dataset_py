import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline

import plotly.io as pio
pio.renderers.default = 'colab'
plotly.offline.init_notebook_mode (connected = True)
dataset=pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")
dataset
dataset.isnull().sum()
dataset.drop(["Serial No."],axis=1,inplace=True)
dataset.rename(columns={"Chance of Admit ":"Chance of Admit"},inplace=True)
dataset.info()
dataset.describe()
plt.figure(figsize=(10,10))
sns.distplot(dataset["GRE Score"],kde=False)
plt.title("Distribution of GRE Score")
plt.show()
plt.figure(figsize=(10,10))
sns.distplot(dataset["TOEFL Score"],kde=False)
plt.title("Distribution of TOEFL Score")
plt.show()
plt.figure(figsize=(10,10))
sns.distplot(dataset["CGPA"],kde=False)
plt.title("Distribution of CGPA")
plt.show()
px.scatter(dataset,y="GRE Score",x="Chance of Admit",color="Chance of Admit",color_continuous_scale=px.colors.sequential.Cividis_r)
px.scatter(dataset,y="TOEFL Score",x="Chance of Admit",color="Chance of Admit",color_continuous_scale=px.colors.sequential.Cividis_r)
px.scatter(dataset,x="University Rating",y="Chance of Admit",color="Chance of Admit",color_continuous_scale=px.colors.sequential.Cividis_r)
px.scatter(dataset,y="CGPA",x="Chance of Admit",color="Chance of Admit",color_continuous_scale=px.colors.sequential.Cividis_r)
plt.figure(figsize=(12,10))
cor=dataset.corr()
sns.heatmap(cor,annot=True)
dataset
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

models=[["Linear Regression : ",LinearRegression()],
        ["Random Forest Regressor : ",RandomForestRegressor()],
        ["Decision Tree Regressor : ",DecisionTreeRegressor()],
        ["K nearest neighbors : ",KNeighborsRegressor(n_neighbors = 2)],
        ["XGB Regressor : ",XGBRegressor()]
        ]

print("Results : ")

for name,model in models:
  regressor=model
  regressor.fit(X_train,y_train)
  y_pred=regressor.predict(X_test)
  print(name, (np.sqrt(mean_squared_error(y_test,y_pred))))
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
a=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()});a.head(10)
a.head()
fig=a.head(25)
fig.plot(kind='bar',figsize=(10,8))
regressor=RandomForestRegressor()
regressor.fit(X,y)
dataset.columns
importance_frame=pd.DataFrame()
importance_frame["features"]=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']
importance_frame["importance"]=regressor.feature_importances_
importance_frame = importance_frame.sort_values(by=['importance'], ascending=True)
px.bar(importance_frame,x="features",y="importance")