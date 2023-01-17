import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly 

import plotly.offline as pyo

from plotly.offline import iplot,plot,init_notebook_mode

import plotly.express as px

import cufflinks as cf

import plotly.graph_objects as go

import seaborn as sns

plt.rc('figure', figsize=(20.0, 10.0))
pyo.init_notebook_mode(connected=True)

cf.go_offline()
import plotly.io as pio

pio.renderers.default = 'colab'
dataset=pd.read_csv("/kaggle/input/playstore-analysis/googleplaystore.csv")
dataset.head()
dataset.isnull().sum()
dataset=dataset.dropna()

dataset=dataset.reset_index(drop=True)
dataset.isnull().sum()
dataset.info()
dataset['Reviews']=dataset["Reviews"].astype(int)
dataset["Size"].unique()
def mb_to_kb(a):

  if a.endswith("M"):

    return float(a[:-1])*1000

  elif a.endswith("k"):

    return float(a[:-1])

  else:

    return a
dataset["Size"]=dataset["Size"].apply(lambda x:mb_to_kb(x))
dataset[dataset["Size"]=="Varies with device"]
rows=dataset[dataset["Size"]=="Varies with device"].index
dataset.drop(rows,inplace=True)
dataset["Installs"].value_counts()
dataset["Installs"]=dataset["Installs"].str[:-1]

dataset["Installs"]=dataset["Installs"].apply(lambda x:x.replace(",",""))
dataset["Installs"]=dataset["Installs"].astype(int)
dataset["Price"].unique()
dataset["Price"]=dataset["Price"].apply(lambda x:x.replace("$",""))

dataset["Price"]=dataset["Price"].astype(float)
dataset["Rating"].between(0,5).sum()
rows=dataset[dataset["Installs"]<dataset["Reviews"]].index

dataset.drop(rows,inplace=True)
dataset.head()
sns.boxplot(data=dataset,orient="h",palette="Set2")
dataset["Reviews"].value_counts()
rows=dataset[dataset["Reviews"]>2000000].index
dataset.drop(rows,inplace=True)
rows=dataset[dataset["Price"]>200].index
dataset.drop(rows,inplace=True)
perc=[.10, .25, .50, .70, .90, .95, .99]

dataset["Installs"].describe(percentiles=perc)
sns.distplot(dataset["Installs"],kde=False)
rows=dataset[dataset["Price"]>500000].index
dataset.drop(rows,inplace=True)
sns.distplot(dataset["Rating"],kde=False)
sns.distplot(dataset["Size"],kde=False)
dataset
plt.figure(figsize=(11,8))

sns.scatterplot(x=dataset["Rating"],y=dataset["Price"],hue=dataset["Rating"])
px.scatter(dataset,x="Rating",y="Size",color="Size",color_continuous_scale=px.colors.sequential.Viridis)
px.scatter(dataset,x="Rating",y="Reviews",color="Size",color_continuous_scale=px.colors.sequential.Viridis)
px.box(dataset,y="Rating",x="Content Rating")
px.box(dataset,y="Rating",x="Category")
dataset.columns
dataset=dataset.reset_index(drop=True)
dataset.drop(["App","Installs","Type","Content Rating",'Last Updated', 'Current Ver',

       'Android Ver'],axis=1,inplace=True)
dataset
X=dataset.iloc[:,1:].values

y=dataset.iloc[:,0].values
X
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [4])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [-4])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

model=regressor.fit(X_train, y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error

print('R2_Score=',r2_score(y_test,y_pred))

print('Root_Mean_Squared_Error(RMSE)=',np.sqrt(mean_squared_error(y_test,y_pred)))


a=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()});a.head(10)
fig=a.head(25)

fig.plot(kind='bar',figsize=(10,8))