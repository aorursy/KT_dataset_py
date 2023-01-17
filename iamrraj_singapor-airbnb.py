# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

'''Seaborn and Matplotlib Visualization'''

import matplotlib                  # 2D Plotting Library

import matplotlib.pyplot as plt

import seaborn as sns              # Python Data Visualization Library based on matplotlib

import geopandas as gpd            # Python Geospatial Data Library

plt.style.use('fivethirtyeight')

%matplotlib inline



'''Plotly Visualizations'''

import plotly as plotly                # Interactive Graphing Library for Python

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)



'''Spatial Visualizations'''

import folium

import folium.plugins



'''NLP - WordCloud'''

import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



'''Machine Learning'''

import sklearn

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
df = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')
#by default head shows us top 5 records

df.head()
df.shape
df.info()
#finding count of total null values in each column

df.isna().sum()
plt.figure(figsize=(20,10))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Fields not needed to our problem

to_drop = ["last_review","reviews_per_month"]

    

# Drop selected fields in place

df.drop(to_drop, inplace=True, axis=1)  
df.dropna(inplace=True)

df.shape

sns.distplot(df["price"])
sns.scatterplot(x='price',y='minimum_nights',data=df)
# sns.countplot(df["neighbourhood_group"])

plt.figure(figsize=(16,10))

sns.countplot(x="neighbourhood_group", data=df)
plt.figure(figsize=(10,6))

sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)

plt.ioff()
import folium

from folium.plugins import HeatMap

m=folium.Map([1.44255,103.79580],zoom_start=11)

HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
ng = df[df.price < 250]

plt.figure(figsize=(10,6))

sns.boxplot(y="price",x ='neighbourhood_group' ,data = ng)

plt.title("neighbourhood_group price distribution < 250")

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

roomdf = df.groupby('room_type').size()/df['room_type'].count()*100

labels = roomdf.index

values = roomdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()

plt.figure(figsize=(10,6))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = df)

plt.title("Room types occupied by the neighbourhood_group")

plt.show()
#catplot room type and price

plt.figure(figsize=(10,6))

sns.catplot(x="room_type", y="price", data=df);

plt.ioff()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()

#let's comeback now to the 'name' column as it will require litte bit more coding and continue to analyze it!



#initializing empty list where we are going to put our name strings

_names_=[]

#getting name strings from the column and appending it to the list

for name in df.name:

    _names_.append(name)

#setting a function that will split those name strings into separate words   

def split_name(name):

    spl=str(name).split()

    return spl

#initializing empty list where we are going to have words counted

_names_for_count_=[]

#getting name string from our list and using split function, later appending to list above

for x in _names_:

    for word in split_name(x):

        word=word.lower()

        _names_for_count_.append(word)

        

#we are going to use counter

from collections import Counter

#let's see top 25 used words by host to name their listing

_top_20_w=Counter(_names_for_count_).most_common()

_top_20_w=_top_20_w[0:20]







#now let's put our findings in dataframe for further visualizations

sub_w=pd.DataFrame(_top_20_w)

sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)

#we are going to use barplot for this visualization

plt.figure(figsize=(10,6))

viz_5=sns.barplot(x='Words', y='Count', data=sub_w)

viz_5.set_title('Counts of the top 20 used words for listing names')

viz_5.set_ylabel('Count of words')

viz_5.set_xlabel('Words')

viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)

df1=df.sort_values(by=['number_of_reviews'],ascending=False).head(1000)

df1.head()
import folium

from folium.plugins import MarkerCluster

from folium import plugins

print("Rooms with the most of reviews")

Long = 103.91492

Lat = 1.32122

mapdf1 = folium.Map([Lat, Long], zoom_start=13)

mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon, label in zip(df1.latitude,df1.longitude,df1.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1
plt.figure(figsize=(10,6))

plt.scatter(df.longitude, df.latitude, c=df.availability_365, cmap='spring', edgecolor='black', linewidth=1\

            , alpha=1)



cbar = plt.colorbar()

cbar.set_label('availability_365')

plt.figure(figsize=(10,6))

sub_6=df[df.price<500]

viz_4=sub_6.plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))

viz_4.legend()

plt.ioff()
#prepare data

df.drop(['name','id','host_name'],axis=1,inplace=True)
'''Encode labels with value between 0 and n_classes-1.'''

le = preprocessing.LabelEncoder()                                            # Fit label encoder

le.fit(df['neighbourhood_group'])

df['neighbourhood_group']=le.transform(df['neighbourhood_group'])    # Transform labels to normalized encoding.



le = preprocessing.LabelEncoder()

le.fit(df['neighbourhood'])

df['neighbourhood']=le.transform(df['neighbourhood'])



le = preprocessing.LabelEncoder()

le.fit(df['room_type'])

df['room_type']=le.transform(df['room_type'])



df.sort_values(by='price',ascending=True,inplace=True)



df.head()
#Train Linear Regression model



lm = LinearRegression()



X = df[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]

y = df['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



lm.fit(X_train,y_train)



predicts = lm.predict(X_test)

error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': predicts.flatten()})

error_airbnb.head()
plt.figure(figsize=(15,8))

sns.regplot('Actual Values','Predicted Values', data=error_airbnb, color='red',marker="+" )
model=LinearRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error, mean_absolute_error

mean_absolute_error(y_test, prediction)

mean_squared_error(y_test, prediction)
plt.figure(figsize=(11,6))

plt.scatter(y_test,prediction,edgecolors='black',c='red',vmin=30,vmax=70)
title=['Pred vs Actual']

fig = go.Figure(data=[

    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),

    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])

])



fig.update_layout(barmode='group')

fig.show()
from sklearn import metrics

mean_sq=metrics.mean_squared_error(y_test,prediction)

RMSE=np.sqrt(mean_sq)

RMSE
from sklearn.metrics import r2_score

r2_score(y_test,prediction)
from sklearn.kernel_ridge import KernelRidge

model1= KernelRidge(alpha=0.029963, kernel='polynomial', degree=1.103746, coef0=5.442672)

model1.fit(X_train,y_train)

prediction1=model1.predict(X_test)

print(mean_absolute_error(y_test, prediction1))

print(r2_score(y_test,prediction1))

print(np.sqrt(metrics.mean_squared_error(y_test,prediction1)))

plt.figure(figsize = (15,15))

sns.heatmap(df.corr(),annot = True,linewidths = 0.5,cmap='cubehelix_r');

plt.savefig('Correlation Heatmap.png')
import xgboost as xgb

model1= xgb.XGBRegressor(n_jobs=-1, n_estimators=849, learning_rate=0.015876, 

                           max_depth=58, colsample_bytree=0.599653, colsample_bylevel=0.287441, subsample=0.154134, seed=18)

model1.fit(X_train,y_train)

prediction2=model1.predict(X_test)

print(mean_absolute_error(y_test, prediction2))

print(r2_score(y_test,prediction2))

print(np.sqrt(metrics.mean_squared_error(y_test,prediction2)))
sns.pairplot(df)

plt.show()
fig = plt.figure(figsize = (15,10))

ax = fig.gca()

df.hist(ax=ax)

plt.show()
df[df.room_type=="Private room"].describe().T
df_rich_hosts=pd.DataFrame(df.iloc[:,0:2])

df_rich_hosts['latitude']=df.iloc[:,3]

df_rich_hosts['longitude']=df.iloc[:,4]

df_rich_hosts['room_type']=df.iloc[:,5]

df_rich_hosts['price']=df.iloc[:,6]

df_rich_hosts.head()
#neighbourhood - availability_365

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')

sns.barplot(x=df.availability_365[:100], y=df.neighbourhood[:100]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')

sns.barplot(x='neighbourhood_group', y="number_of_reviews", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
ax = sns.violinplot(x="room_type", y="price",

                    data=df[df.price > 500],

                    scale="width", palette="Set3")

sns.kdeplot(df['price'])

plt.xlabel('Values')

plt.ylabel('Frequency')

plt.title('Price Kde Plot')

plt.show()
neighbourhood_group - calculated_host_listings_count

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')

sns.barplot(x='neighbourhood_group', y="calculated_host_listings_count", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()

#neighbourhood_group - reviews_per_month

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')

sns.barplot(x='neighbourhood_group', y="reviews_per_month", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
neighbourhood_group - calculated_host_listings_count

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')

sns.barplot(x='neighbourhood_group', y="calculated_host_listings_count", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()

sns.lineplot(x='calculated_host_listings_count',y='price',data=df)

plt.show()

#neighbourhood_group - availability_365

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')

sns.barplot(x='neighbourhood_group', y="availability_365", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='availability_365',y='price',data=df)

plt.show()
df.price.describe().T
labels = df.room_type.value_counts().index

colors = ['orange','yellow','red']

explode = [0,0,0]

sizes = df.room_type.value_counts().values



plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Airbnb According to Room Type',color = 'blue',fontsize = 15)

plt.show()
#room_type - minimum_nights

result = df.groupby(["room_type"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')

sns.barplot(x='room_type', y="minimum_nights", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()

sns.lineplot(x='minimum_nights',y='price',data=df)

plt.show()

#room_type - number_of_reviews

result = df.groupby(["room_type"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')

sns.barplot(x='room_type', y="number_of_reviews", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()

sns.lineplot(x='number_of_reviews',y='price',data=df)

plt.show()

#room_type - reviews_per_month

result = df.groupby(["room_type"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')

sns.barplot(x='room_type', y="reviews_per_month", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()

#room_type - calculated_host_listings_count

result = df.groupby(["room_type"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')

sns.barplot(x='room_type', y="calculated_host_listings_count", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#room_type - availability_365

result = df.groupby(["room_type"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')

sns.barplot(x='room_type', y="availability_365", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()

#neighbourhood - price

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['price'].aggregate(np.median).reset_index().sort_values('price')

sns.barplot(x=df.price[:50], y=df.neighbourhood[:50]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df_rich_hosts.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="yellow").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
