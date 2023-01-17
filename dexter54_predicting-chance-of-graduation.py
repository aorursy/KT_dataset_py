import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objects as go
data=pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head()
data.drop("Serial No.",inplace=True,axis=1)
data.head()
data.dtypes
data[["GRE Score","TOEFL Score","University Rating","Research"]]=data[["GRE Score","TOEFL Score","University Rating","Research"]].astype(float)
sns.pairplot(data)
data.head()
f,ax=plt.subplots(1,3,figsize=(20,10))

sns.scatterplot(x=data["TOEFL Score"],y=data["GRE Score"],hue=data["Chance of Admit "],s=100,ax=ax[0])

sns.scatterplot(x=data["TOEFL Score"],y=data["GRE Score"],hue=data["Research"],s=100,ax=ax[1])

sns.scatterplot(x=data["TOEFL Score"],y=data["GRE Score"],hue=data["University Rating"],s=100,ax=ax[2])

ax[0].title.set_text("GRE vs TOEFL Score with respect to Chance of Admit")

ax[1].title.set_text("GRE vs TOEFL Score with respect to Research")

ax[2].title.set_text("GRE vs TOEFL Score with respect to University Rating")
x=data["University Rating"]

trace1=go.Histogram(

    x=x,

    opacity=0.75,

    name="Universit Rating",

    marker=dict(color='#330C73'))

    

    

    

Data=[trace1]

layout=go.Layout(barmode="overlay",#iç içe geçmek 

                title="Number of University Ratings",

                xaxis=dict(title="University Rating"),

                yaxis=dict(title="Count")

    )

fig=go.Figure(data=Data,layout=layout)

fig.update_layout(

    title={

        'y':0.9,

        'x':0.5

        

        

        

        

    }





)

iplot(fig)
fig,axes=plt.subplots(ncols=2,nrows=1,figsize=(10,10))



sns.boxplot(data=data,y="Chance of Admit ",x="University Rating",ax=axes[0])

sns.boxplot(data=data,y="Chance of Admit ",x="Research",ax=axes[1])
mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))

f, ax = plt.subplots(figsize=(7, 5))

ax = sns.heatmap(data.corr(), mask=mask, square=True,cmap="PiYG",annot=True,fmt=".1f",vmin=0,vmax=1)
import statsmodels.api as sm

import statsmodels.formula.api as smf

X=data.iloc[:,:-1]

y=data.iloc[:,-1]

X=sm.add_constant(X)

model=sm.OLS(y,X).fit()

predictions=model.predict(X)

print(model.summary())
X=data["CGPA"].values.reshape(-1,1)

y=data.iloc[:,-1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

lg=LinearRegression()

lg.fit(X_train,y_train)
plt.scatter(X_train,y_train,color="r")

plt.plot(X_train,lg.predict(X_train))

plt.title("Chance of Admit vs CGPA (Training Set)")

plt.xlabel("CGPA")

plt.ylabel("Chance of Admit")
plt.scatter(X_test,y_test,color="r")

plt.plot(X_test,lg.predict(X_test))

plt.title("Chance of Admit vs CGPA (Test Set)")

plt.xlabel("CGPA")

plt.ylabel("Chance of Admit")

plt.show
df=pd.DataFrame(zip(y_test,lg.predict(X_test)),columns=["True Result","Prediction"])

df.head(10)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

evaluation=pd.DataFrame(columns=["Regression Type","Mean Absolute Error","Mean Squared Error","R^2"])

r_squared=r2_score(y_test,lg.predict(X_test))



evaluation.loc[0]=["Simple Linear Regression",mean_absolute_error(y_test,lg.predict(X_test)),mean_squared_error(y_test,lg.predict(X_test)),r_squared]
evaluation.head()
X=data.iloc[:,:-1].values

y=data.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lg=LinearRegression()

lg.fit(X_train,y_train)
pred=lg.predict(X_test)
r_squared=r2_score(y_test,pred)

evaluation.loc[1]=["Multiple Linear Regression",mean_absolute_error(y_test,pred),mean_squared_error(y_test,pred),r_squared]
evaluation.head()
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

sc_y=StandardScaler()

X_ScaledTrain=sc_X.fit_transform(X_train)

X_ScaledTest=sc_X.transform(X_test)

y_ScaledTrain=sc_y.fit_transform(y_train.reshape(-1,1))

y_ScaledTest=sc_y.transform(y_test.reshape(-1,1))
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

grid_rbf= GridSearchCV(SVR(), param_grid, refit = True, verbose = 3) 

grid_rbf.fit(X_ScaledTrain,y_ScaledTrain.ravel())
pred_rbf=grid_rbf.predict(X_ScaledTest)
RSquared_rbf=r2_score(y_ScaledTest,pred_rbf)

evaluation.loc[2]=["Support Vector Regression (RBF)",mean_absolute_error(y_ScaledTest,pred_rbf),mean_squared_error(y_ScaledTest,pred_rbf),RSquared_rbf]
evaluation.head()
from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor()

regressor.fit(X_train,y_train)
pred=regressor.predict(X_test)
r_squared=r2_score(y_test,pred)

evaluation.loc[5]=[" Decision Tree Regression",mean_absolute_error(y_test,pred),mean_squared_error(y_test,pred),r_squared]
evaluation.head()
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 200)

regressor.fit(X_train,y_train)
pred=regressor.predict(X_test)
r_squared=r2_score(y_test,pred)

evaluation.loc[6]=["Random Forest Regression",mean_absolute_error(y_test,pred),mean_squared_error(y_test,pred),r_squared]
evaluation.head()