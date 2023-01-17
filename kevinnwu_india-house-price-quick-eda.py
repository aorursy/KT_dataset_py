import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.figure_factory as ff

%matplotlib inline 
train = pd.read_csv('../input/house-price-prediction-challenge/train.csv')

x_test = pd.read_csv('../input/house-price-prediction-challenge/test.csv')

y_test = pd.read_csv('../input/house-price-prediction-challenge/sample_submission.csv')
train.head()
train.isnull().sum()
train.info()
train.describe()
plt.figure(figsize=(12,10))

sns.heatmap(train.corr(), annot=True)
fig = px.histogram(train, x="TARGET(PRICE_IN_LACS)",marginal="rug")

fig.show()
fig = px.bar(x=train["BHK_NO."].unique(), y=train["BHK_NO."].value_counts())

fig.show()
fig = px.histogram(train, x="SQUARE_FT",marginal="rug")

fig.show()
import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
map = folium.Map(location=[22.00,78.00], tiles='cartodbpositron', zoom_start=6)





for i in range(0,len(train)):

    Circle(

        location=[train.iloc[i]['LONGITUDE'], train.iloc[i]['LATITUDE']],

        radius=100,

        color='blue').add_to(map)



# Display the map

map
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
train
x_train = train.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS', 'LATITUDE', 'LONGITUDE', 'TARGET(PRICE_IN_LACS)'], axis=1)

y_train = train['TARGET(PRICE_IN_LACS)']
lm.fit(x_train, y_train)
print(lm.intercept_)
lm.coef_

pd.DataFrame(lm.coef_, x_train.columns, columns=['Coeff'])
x_test = x_test.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS', 'LATITUDE', 'LONGITUDE'], axis=1)
predictions = lm.predict(x_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,predictions))
pd.DataFrame(predictions).to_csv('submission.csv')