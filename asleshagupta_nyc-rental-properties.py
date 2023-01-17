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
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df.head()
df.info()
df.isnull().sum()
df.drop(['name','host_name','last_review'], axis=1, inplace=True)

df.head()
df.fillna({'reviews_per_month':0}, inplace=True)

df.isnull().sum()
df.describe().T
label = df['neighbourhood_group'].unique()

sizes = df['neighbourhood_group'].value_counts()

colors = ['lightcoral','indianred', 'tomato','orangered', 'salmon']

fig, ax = plt.subplots(1,1,figsize=(10,10))

plt.pie(sizes,explode=[0.009,0.006,0,0,0], labels = label, shadow = False, labeldistance = 1.1,autopct='%1.3f%%', startangle = 90, colors=colors)

ax.set_title('Listings share as per Neighbourhood')

plt.show()

#color2 = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

fig2, ax2 = plt.subplots(1,1, figsize=(10,10))

sns.countplot('neighbourhood_group',data=df, order=df['neighbourhood_group'].value_counts().index)

ax2.set_title('Number of Listing - Bar Chart')
import folium

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.3:'green',0.6:'yellow',1.0:'red'}).add_to(m)

display(m)

df2 = df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().copy()

df2.head()

df3 = df[['neighbourhood_group']]

# df2.reset_index()

# df2['price'][1]
m2=folium.Map([40.7128,-74.0060],zoom_start=11, titles = "HeatMap of Price")

HeatMap(df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(),radius=10,gradient={0.25:'green',0.6:'yellow',0.8:'blue',1.0:'red'}).add_to(m2)

# folium.CircleMarker(df2[['latitude','longitude'][1]], popup = 'some').add_to(m2)

# m2.add_child(folium.ClickforMarker(popup='Awesome'))

# m2.add_child(folium.ClickForMarker(df2[['latitude','longitude'][1]], popup='some'))

# MarkerCluster(df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(), popups = df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(), overlay=True, control=True, show=True).add_to(m2)

# BoatMarker(df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(),radius=10,gradient={0.25:'green',0.6:'yellow',0.8:'blue',1.0:'red'}).add_to(m2)

# width = df[['latitude']], height = df[['longitutde']], radius =

display(m2)
import plotly.graph_objects as go

# df['text'] = df['neighbourhood'] + ':' + df['price'] 

# + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

fig = go.Figure(data=go.Scattergeo(

        lon = df['longitude'],

        lat = df['latitude'],

        mode = 'markers',

#         text = df['text'],

        marker_color = df['price'],

#         marker_size = df['price']

        ))



fig.update_layout(

        title = 'Price Scatter Plot',

        geo_scope='usa',

    )

fig.show()
m3=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(df[['latitude','longitude','availability_365']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(),radius=10,gradient={0.25:'green',0.6:'yellow',0.8:'blue',1.0:'red'}).add_to(m3)

display(m3)

#0.25:'green',0.6:'yellow',0.8:'orange',1.0:'red'
#Next we will check price distribution in these neighbourhood groups

#creating a sub-dataframe with no extreme values / less than 500

sub_df=df[df.price < 300]

fig3 = plt.subplots(figsize=(10,10))

viz=sns.violinplot(data=sub_df, x='neighbourhood_group', y='price', scale="count", hue = 'room_type')

viz.set_title('Distribution of prices for each neighberhood_group')
df_cat = df.groupby(['neighbourhood_group','room_type'])['price'].mean().reset_index()

df_cat
#Scatter plot between availability and price

sx = df['availability_365']

sy = df['price']

plt.scatter(sx,sy,alpha=0.5)

plt.show()
#transform data

from sklearn import preprocessing



enc = preprocessing.LabelEncoder()

enc.fit(df['neighbourhood_group'])

df['neighbourhood_group']=enc.transform(df['neighbourhood_group'])    



enc = preprocessing.LabelEncoder()

enc.fit(df['neighbourhood'])

df['neighbourhood']=enc.transform(df['neighbourhood'])



enc = preprocessing.LabelEncoder()

enc.fit(df['room_type'])

df['room_type']=enc.transform(df['room_type'])



df.drop(['id', 'host_id'], axis = 1, inplace=True)
plt.figure(figsize=(30, 30))

sns.pairplot(df, height=3, diag_kind="hist")
df.head()
from sklearn.preprocessing import StandardScaler

col_to_scale = ['neighbourhood','room_type','availability_365','latitude','longitude','number_of_reviews','calculated_host_listings_count','minimum_nights','neighbourhood_group','reviews_per_month']

df[col_to_scale] = StandardScaler().fit_transform(df[col_to_scale])

df.head()
#Regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

lm = LinearRegression()

#'minimum_nights',,'id''neighbourhood_group','reviews_per_month',

X = df[['neighbourhood','room_type','availability_365','latitude','longitude','number_of_reviews','calculated_host_listings_count']]

y = df['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)



lm.fit(X_train,y_train)



from sklearn import metrics

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pred = lm.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, pred)),

        r2_score(y_test,pred) * 100,

        mean_absolute_error(y_test,pred)

        ))