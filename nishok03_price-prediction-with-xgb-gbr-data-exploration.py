# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly as plotly

import seaborn as sns

from sklearn import preprocessing

import geopandas as gpd

%matplotlib inline
from plotly import __version__

import plotly.offline as py 

from plotly.offline import init_notebook_mode, plot

init_notebook_mode(connected=True)

from plotly import tools

import plotly.graph_objs as go

import plotly.express as px

import folium

from folium.plugins import MarkerCluster

from folium import plugins

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df1=df.sort_values(by=['number_of_reviews'],ascending=False).head(1000)
df2=df.sort_values(by=['price'],ascending=False).head(1000)
print('Rooms with the most number of reviews')

Long=-73.80

Lat=40.80

mapdf1=folium.Map([Lat,Long],zoom_start=10,)



mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon,label in zip(df1.latitude,df1.longitude,df1.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1
print('Most Expensive rooms')

Long=-73.80

Lat=40.80

mapdf1=folium.Map([Lat,Long],zoom_start=10,)



mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon,label in zip(df2.latitude,df2.longitude,df2.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1
plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=df)

plt.show()
df3=df.groupby(['neighbourhood_group']).mean()
df3.drop(['latitude', 'longitude','host_id','id'],axis=1)
df4=df.groupby(['neighbourhood_group','neighbourhood']).mean()
r1=df4.loc['Bronx'].number_of_reviews.sum().round()

r2=df4.loc['Brooklyn'].number_of_reviews.sum().round()

r3=df4.loc['Manhattan'].number_of_reviews.sum().round()

r4=df4.loc['Queens'].number_of_reviews.sum().round()

r5=df4.loc['Staten Island'].number_of_reviews.sum().round()
abcd=df['neighbourhood_group'].value_counts()

dfabcd=pd.DataFrame(abcd)

dfabcd.reset_index(inplace=True)

reviews = [r1,r2,r3,r4,r5]

review = pd.DataFrame(data=reviews,index=dfabcd['index'],columns=['values'],)

review.reset_index(inplace=True)







trace10 = go.Bar(x=review['index'],y=review['values'],marker=dict(color=['Blue','Red','Green','Black','Purple']),width=0.4)



data=[trace10]

layout = go.Layout(title='Number of reviews by Neighbourhood',height=400,width=800)

fig= go.Figure(data=data,layout=layout)

py.iplot(fig)

r1=df4.loc['Bronx'].reviews_per_month.mean()

r2=df4.loc['Brooklyn'].reviews_per_month.mean()

r3=df4.loc['Manhattan'].reviews_per_month.mean()

r4=df4.loc['Queens'].reviews_per_month.mean()

r5=df4.loc['Staten Island'].reviews_per_month.mean()



rev = [r1,r2,r3,r4,r5]



rev_per_month = pd.DataFrame(data=rev,columns=['values'],index=dfabcd['index'])



rev_per_month.reset_index(inplace=True)





trace2 = go.Scatter(x=rev_per_month['index'],y=rev_per_month['values'],marker=dict(color=['Blue','Red','Green','Black','Purple']))

data=[trace2]

layout = go.Layout(title='Average Reviews per month per place by Neighbourhood',height=400,width=800)

fig= go.Figure(data=data,layout=layout,)

py.iplot(fig)

df['room_type'].value_counts()
df5 = df.groupby(['neighbourhood_group','room_type']).mean()
room_types_neighbourhoods=df5.drop(['id','host_id','latitude','longitude','number_of_reviews','reviews_per_month'],axis=1)
room_types_neighbourhoods
fig = px.scatter_matrix(room_types_neighbourhoods,height=1000,width=900,color="minimum_nights")

fig.update_traces(diagonal_visible=False)

fig.show()

df6 = df.groupby(['room_type']).mean()

room_types =df6.drop(['id','host_id','latitude','longitude','number_of_reviews','reviews_per_month'],axis=1)
room_types
df7 = df.groupby(['neighbourhood_group','room_type'])['id'].agg('count')
rmtng = pd.DataFrame(df7)
rmtng.reset_index(inplace=True)
Bronx = rmtng[rmtng['neighbourhood_group']=='Bronx']

Brooklyn = rmtng[rmtng['neighbourhood_group']=='Brooklyn']

Manhattan = rmtng[rmtng['neighbourhood_group']=='Manhattan']

Queens = rmtng[rmtng['neighbourhood_group']=='Queens']

StatenIsland = rmtng[rmtng['neighbourhood_group']=='Staten Island']



df8 = df.groupby(['room_type']).count()

rooms = df8.drop(['host_name','name','host_id','latitude','longitude','number_of_reviews','reviews_per_month','neighbourhood_group','neighbourhood','price','minimum_nights','calculated_host_listings_count','availability_365','last_review'],axis=1)

rooms.reset_index(inplace=True)

trace1=go.Bar(x=Bronx['room_type'],y=Bronx['id'],name='Bronx')

trace2=go.Bar(x=Brooklyn['room_type'],y=Brooklyn['id'],name='Brooklyn')

trace3=go.Bar(x=Manhattan['room_type'],y=Manhattan['id'],name='Manhattan')

trace4=go.Bar(x=Queens['room_type'],y=Queens['id'],name='Queens')

trace5=go.Bar(x=StatenIsland['room_type'],y=StatenIsland['id'],name='StatenIsland')

trace6=go.Bar(x=rooms['room_type'],y=rooms['id'],name='Total')



titles=['Room types - Bronx',

        'Room types - Brooklyn',

        'Room types - Manhattan',

        'Room types - Queens',

        'Room types - StatenIsland',

        'Room types - All Neighbourhoods']



fig=plotly.subplots.make_subplots(rows=2,cols=3,subplot_titles=titles,)

fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,2)

fig.append_trace(trace3,1,3)

fig.append_trace(trace4,2,1)

fig.append_trace(trace5,2,2)

fig.append_trace(trace6,2,3)

fig['layout'].update(height=1000,width=1000,paper_bgcolor='white')

py.iplot(fig,filename='rmtypeplot')
rmtng2 = df.groupby(['neighbourhood_group','neighbourhood'])['price'].agg('mean')
rmtng1 = pd.DataFrame(rmtng2)

rmtng1.reset_index(inplace=True)
Bronx = rmtng1[rmtng1['neighbourhood_group']=='Bronx']

Brooklyn = rmtng1[rmtng1['neighbourhood_group']=='Brooklyn']

Manhattan = rmtng1[rmtng1['neighbourhood_group']=='Manhattan']

Queens = rmtng1[rmtng1['neighbourhood_group']=='Queens']

StatenIsland = rmtng1[rmtng1['neighbourhood_group']=='Staten Island']
Bronx1=Bronx.sort_values(by=['price'],ascending=False).head(10)

Brooklyn1=Brooklyn.sort_values(by=['price'],ascending=False).head(10)

Manhattan1=Manhattan.sort_values(by=['price'],ascending=False).head(10)

Queens1=Queens.sort_values(by=['price'],ascending=False).head(10)

StatenIsland1=StatenIsland.sort_values(by=['price'],ascending=False).head(10)
trace1=go.Scatter(x=Bronx1['neighbourhood'],y=Bronx1['price'],marker=dict(color="crimson", size=12),mode="markers",name="Bronx",)



trace2=go.Scatter(x=Brooklyn1['neighbourhood'],y=Brooklyn1['price'],marker=dict(color="blue", size=12),mode="markers",name="Brooklyn",)



trace3=go.Scatter(x=Manhattan1['neighbourhood'],y=Manhattan1['price'],marker=dict(color="purple", size=12),mode="markers",name="Manhattan",)



trace4=go.Scatter(x=Queens1['neighbourhood'],y=Queens1['price'],marker=dict(color="black", size=12),mode="markers",name="Queens",)



trace5=go.Scatter(x=StatenIsland1['neighbourhood'],y=StatenIsland1['price'],marker=dict(color="red", size=12),mode="markers",name="StatenIsland",)



data = [trace1,trace2,trace3,trace4,trace5]



titles=['Most Pricey neighbourhoods-Bronx',

        'Most Pricey neighbourhoods-Brooklyn',

        'Most Pricey neighbourhoods-Manhattan',

        'Most Pricey neighbourhoods-Queens',

        'Most Pricey neighbourhoods-StatenIsland']



fig =plotly.subplots.make_subplots(rows=3,cols=2,subplot_titles=titles)





fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,2)

fig.append_trace(trace3,2,1)

fig.append_trace(trace4,2,2)

fig.append_trace(trace5,3,1)





fig['layout'].update(height=1200,width=1000,paper_bgcolor='white')



py.iplot(fig,filename='pricetypeplot')
pnd2 =df.groupby(['neighbourhood','neighbourhood_group']).agg('count')

pnd2.reset_index(inplace=True)

pnd2.set_index(['neighbourhood_group'],inplace=True)

pnd2.sort_index(inplace=True)

pnd2.drop(['name',

           'host_id',

           'host_name',

           'latitude','longitude',

           'room_type','price','minimum_nights',

           'number_of_reviews','last_review','reviews_per_month',

           'calculated_host_listings_count','availability_365'],

            axis=1,inplace=True)

Bronx = pnd2[pnd2.index=='Bronx']

Brooklyn = pnd2[pnd2.index=='Brooklyn']

Manhattan = pnd2[pnd2.index=='Manhattan']

Queens = pnd2[pnd2.index=='Queens']

StatenIsland = pnd2[pnd2.index=='Staten Island']
Bronx2=Bronx.sort_values(by='id',ascending=False).head(10)

Brooklyn2=Brooklyn.sort_values(by='id',ascending=False).head(10)

Manhattan2=Manhattan.sort_values(by='id',ascending=False).head(10)

Queens2=Queens.sort_values(by='id',ascending=False).head(10)

StatenIsland2=StatenIsland.sort_values(by='id',ascending=False).head(10)
trace1=go.Pie(labels=Bronx2['neighbourhood'], values=Bronx2['id'], name="Bronx Neighbourhoods",showlegend=False)

trace2=go.Pie(labels=Brooklyn2['neighbourhood'], values=Brooklyn2['id'], name="Brooklyn Neighbourhoods",showlegend=False)

trace3=go.Pie(labels=Manhattan2['neighbourhood'], values=Manhattan2['id'], name="Manhattan Neighbourhoods",showlegend=False)

trace4=go.Pie(labels=Queens2['neighbourhood'], values=Queens2['id'], name="Queens Neighbourhoods",showlegend=False)

trace5=go.Pie(labels=StatenIsland2['neighbourhood'], values=StatenIsland2['id'], name="StatenIsland Neighbourhoods",showlegend=False)





titles=['Popular neighbourhoods-Bronx',

        'Popular neighbourhoods-Brooklyn',

        'Popular neighbourhoods-Manhattan',

        'Popular neighbourhoods-Queens',

        'Popular neighbourhoods-StatenIsland']







fig =plotly.subplots.make_subplots(rows=3,cols=2,subplot_titles=titles,specs=[[{"type": "domain"}, {"type": "domain"}],

                                                                             [{"type": "domain"}, {"type": "domain"}],

                                                                             [{"type": "domain"}, {"type": "domain"}]])

                                                                     



fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,2)

fig.append_trace(trace3,2,1)

fig.append_trace(trace4,2,2)

fig.append_trace(trace5,3,1)





fig['layout'].update(height=1000,width=800,paper_bgcolor='white')



py.iplot(fig,filename='pricetypeplot')
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df1 = df



df1.drop(['name','id','host_name','last_review'],axis=1,inplace=True)

df1['reviews_per_month']=df1['reviews_per_month'].replace(np.nan, 0)



le = preprocessing.LabelEncoder()

le.fit(df1['neighbourhood_group'])    

df1['neighbourhood_group']=le.transform(df1['neighbourhood_group'])



le = preprocessing.LabelEncoder()

le.fit(df1['neighbourhood'])

df1['neighbourhood']=le.transform(df1['neighbourhood'])



le = preprocessing.LabelEncoder()

le.fit(df1['room_type'])

df1['room_type']=le.transform(df1['room_type'])



df1.sort_values(by='price',ascending=True,inplace=True)



df1.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression
lm = LinearRegression()
X = df1[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]

y = df1['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
lm.fit(X_train,y_train)
predicts = lm.predict(X_test)
from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error
print("Root mean squared error is:")

np.sqrt(metrics.mean_squared_error(y_test,predicts))
print('r2 score is:')

r2 = r2_score(y_test,predicts)

r2*100
print("Mean absolute error is:")

mean_absolute_error(y_test,predicts)
error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': predicts.flatten()})

error_diff1 = error_diff.head(20)
error_diff1.head(5)
title=['Pred vs Actual']

fig = go.Figure(data=[

    go.Bar(name='Predicted', x=error_diff1.index, y=error_diff1['Predicted Values']),

    go.Bar(name='Actual', x=error_diff1.index, y=error_diff1['Actual Values'])

])



fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(16,8))

sns.regplot(predicts,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.show()
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)
GBoost.fit(X_train,y_train)
predict = GBoost.predict(X_test)
print("Root mean squared error is:")

np.sqrt(metrics.mean_squared_error(y_test,predict))
print('r2 score is:')

r2 = r2_score(y_test,predict)

r2*100
print("Mean absolute error is:")

mean_absolute_error(y_test,predict)
error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': predict.flatten()})

error_diff1 = error_diff.head(20)
error_diff1.head()
title='Pred vs Actual'

fig = go.Figure(data=[

    go.Bar(name='Predicted', x=error_diff1.index, y=error_diff1['Predicted Values']),

    go.Bar(name='Actual', x=error_diff1.index, y=error_diff1['Actual Values'])

])

fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(16,8))

sns.regplot(predict,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Gradient Boosted Regressor model Predictions")

plt.show()
import xgboost

import warnings 

warnings.simplefilter(action='ignore')
xgb = xgboost.XGBRegressor(n_estimators=310,learning_rate=0.1,objective='reg:squarederror')

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
print("Root mean squared error is:")

np.sqrt(metrics.mean_squared_error(y_test,xgb_pred))
print('r2 score is:')

r2 = r2_score(y_test,xgb_pred)

r2*100
print("Mean absolute error is:")

mean_absolute_error(y_test,xgb_pred)
error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': xgb_pred.flatten()})

error_diff1 = error_diff.head(20)
error_diff1.head()
title='Pred vs Actual'

fig = go.Figure(data=[

    go.Bar(name='Predicted', x=error_diff1.index, y=error_diff1['Predicted Values']),

    go.Bar(name='Actual', x=error_diff1.index, y=error_diff1['Actual Values'])

])

fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(16,8))

sns.regplot(xgb_pred,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Xgboost Regressor Predictions")

plt.show()


