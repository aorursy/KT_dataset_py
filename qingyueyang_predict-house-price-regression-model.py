import os

print(os.listdir("../input"))
# Import the libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from learntools.core import *

kc_data = pd.read_csv('../input/kc_house_data.csv')

kc_data.head()
# Information about the dataset

kc_data.info()
# Statistical summary of the dataset

kc_data.describe().transpose()
import plotly.plotly as py

import plotly.graph_objs as go

import plotly.offline as ply

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from plotly import tools

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style= "whitegrid")



corr_mat = kc_data.corr()

plt.figure(figsize=(30,15))

sns.heatmap(corr_mat, cmap = 'BrBG', linecolor = 'white', linewidth = 1, annot=True)
plt.figure(figsize=(12,5))

sns.distplot(kc_data['price'])
fig1 = go.Scattergl(x=kc_data['sqft_living'], y=kc_data['price'], mode='markers', name='sqft_living')

fig2 = go.Scattergl(x=kc_data['bedrooms'], y=kc_data['price'], mode = 'markers', name = 'bedrooms')

fig3 = go.Scattergl(x=kc_data['bathrooms'], y=kc_data['price'],mode = 'markers', name = 'bathrooms')

fig4 = go.Scattergl(x=kc_data['grade'], y=kc_data['price'],mode = 'markers', name = 'grade')

fig5 = go.Scattergl(x=kc_data['yr_built'], y=kc_data['price'],mode = 'markers', name = 'yr_built')

fig6 = go.Scattergl(x=kc_data['lat'], y=kc_data['price'],mode = 'markers', name = 'lat')

fig = tools.make_subplots(rows=2, cols=3, subplot_titles=('sqft_living vs Price', 'bedrooms vs Price',

'bathrooms vs Price', 'grade vs Price', 'yr_built vs price', 'lat vs price'))

fig.append_trace(fig1, 1, 1)

fig.append_trace(fig2, 1, 2)

fig.append_trace(fig3, 1, 3)

fig.append_trace(fig4, 2, 1)

fig.append_trace(fig5, 2, 2)

fig.append_trace(fig6, 2, 3)

fig['layout'].update(height=800, width=800, title='Price Subplots')

ply.iplot(fig)
kc_df = kc_data.drop(kc_data[kc_data["bedrooms"]>10].index )
from sklearn.model_selection import train_test_split

y = kc_df.price

features = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built', 'lat']

X = kc_df[features]

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)



from sklearn.linear_model import LinearRegression



kc_lrmodel = LinearRegression()

kc_lrmodel.fit(X_train, y_train)

# Predicting the Test set results

y_lrpred = kc_lrmodel.predict(X_test)

from sklearn.svm import SVR



kc_svrmodel = SVR(kernel='rbf')

kc_svrmodel.fit(X_train, y_train)

y_svrpred = kc_svrmodel.predict(X_test)
from sklearn.neighbors import KNeighborsRegressor



kc_knnmodel = KNeighborsRegressor(n_neighbors=1)

kc_knnmodel.fit(X_train,y_train)

y_knnpred = kc_knnmodel.predict(X_test)
from sklearn.ensemble import RandomForestRegressor



kc_rfmodel = RandomForestRegressor(n_estimators=20, random_state = 0)

kc_rfmodel.fit(X_train, y_train)

y_rfpred = kc_rfmodel.predict(X_test)

from xgboost import XGBRegressor



kc_xgbmodel = XGBRegressor()

kc_xgbmodel.fit(X_train, y_train)

y_xgbpred = kc_xgbmodel.predict(X_test) 
# Calculate Adjusted R Squared Value

from sklearn import metrics

lr_R = metrics.r2_score(y_test,y_lrpred)

lr_a_R = 1 - (1-lr_R)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print('Adjusted R Squared Value for Linear Regression: ', round(lr_a_R, 3) )



svr_R = metrics.r2_score(y_test,y_svrpred)

svr_a_R = 1 - (1-svr_R)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print('Adjusted R Squared Value for SVR: ', round(svr_a_R, 3) )



rf_R = metrics.r2_score(y_test,y_rfpred)

rf_a_R = 1 - (1-rf_R)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print('Adjusted R Squared Value for Random Forest: ', round(rf_a_R, 3) )



knn_R = metrics.r2_score(y_test,y_knnpred)

knn_a_R = 1 - (1-knn_R)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print('Adjusted R Squared Value for KNN: ', round(knn_a_R, 3) )



xgb_R = metrics.r2_score(y_test,y_xgbpred)

xgb_a_R = 1 - (1-xgb_R)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print('Adjusted R Squared Value for XGBoost: ', round(xgb_a_R, 3) )

columns = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built', 'lat']

sample = pd.DataFrame([[4, 3.25, 3360, 10, 1994, 47.70]],

                        columns = columns )

customer = sc_X.transform(sample)



lrpredictor = kc_lrmodel.predict(customer)

print('Prediction by Linear Regression is', lrpredictor)



svrpredictor = kc_svrmodel.predict(customer)

print('Prediction by SVR is',svrpredictor)



rfpredictor = kc_rfmodel.predict(customer)

print('Prediction by Random Forest is', rfpredictor)



knnpredictor = kc_knnmodel.predict(customer)

print('Prediction by KNN is',knnpredictor)



xgbpredictor = kc_xgbmodel.predict(customer)

print('Prediction by XGBoost is',xgbpredictor)