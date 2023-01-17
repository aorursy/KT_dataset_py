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
'''Importing Data Manipulation Modules'''

import numpy as np                 # Linear Algebra

import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)



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
data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
print('Rows     :',data.shape[0])

print('Columns  :',data.shape[1])

print('\nFeatures :\n     :',data.columns.tolist())

print('\nMissing values    :',data.isnull().values.sum())

print('\nUnique values :  \n',data.nunique())
data.shape
total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum())*100)/data.isnull().count().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
#make a list of the variables that contain missing values

vars_with_na=[var for var in data.columns if data[var].isnull().sum()>1]



#print the variable name and the percentage of missing values 

for var in vars_with_na:

    print(var,np.round(data[var].isnull().mean(),3),'% missing values')
data.describe().T
data.corr().style.background_gradient(cmap='coolwarm')

#No strong correlation except number_of_reviews vs reviews_per_month
def analyse_na_value(df,var):

    df=df.copy()

    

    #Let's make a variable that indicates 1 if the observation was missing or Zero otherwise 

    df[var]=np.where(df[var].isnull(),1,0)

    

    #Let's calculate the mean SalePrice where the information is missing or present 

    df.groupby(var)['price'].median().plot.bar()

    plt.title(var)

    plt.show()

    

for var in vars_with_na:

    analyse_na_value(data,var)
f,ax=plt.subplots(1,2,figsize=(18,8))

data['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.05,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Share of Neighborhood')

ax[0].set_ylabel('Neighborhood Share')

sns.countplot('neighbourhood_group',data=data,ax=ax[1],order=data['neighbourhood_group'].value_counts().index)

ax[1].set_title('Share of Neighborhood')

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group)

plt.ioff()
import folium

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(data[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
plt.figure(figsize=(10,6))

sns.distplot(data[data.neighbourhood_group=='Manhattan'].price,color='maroon',hist=False,label='Manhattan')

sns.distplot(data[data.neighbourhood_group=='Brooklyn'].price,color='black',hist=False,label='Brooklyn')

sns.distplot(data[data.neighbourhood_group=='Queens'].price,color='green',hist=False,label='Queens')

sns.distplot(data[data.neighbourhood_group=='Staten Island'].price,color='blue',hist=False,label='Staten Island')

sns.distplot(data[data.neighbourhood_group=='Long Island'].price,color='lavender',hist=False,label='Long Island')

plt.title('Borough wise price destribution for price<2000')

plt.xlim(0,2000)

plt.show()
plt.figure(figsize=(14,8))

sns.distplot(data.minimum_nights).set_yscale('log')

plt.title('Minimum no. of nights distribution')

plt.show()
## setting style for our plots

sns.set(style="white", palette="spring", color_codes=True)

## ignore -- f, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True) -- ignore ##



## figure size with 10 width and 5 height

plt.figure(figsize=(10, 5))

## create dataframe "df1" with all the neighbourhood of Brooklyn and their price

df1 = data[data.neighbourhood_group == "Brooklyn"][["neighbourhood","price"]]

## lets take mean of all the prices of neighbouhood

d = df1.groupby("neighbourhood").mean()

## distplot -> distribution plot

## axlabel == xlabel

## kde_kws -> kernel density estimate keyword arguments -> color="black"

## hist_kws -> histogram keyword arguments -> histogram type = step

sns.distplot(d,color='r',axlabel ="Price Distribution in Brooklyn",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3});

plt.ioff()

plt.plot();
plt.figure(figsize=(10,6));

sub_6=data[data.price<500];

viz_4=sub_6.plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10));

viz_4.legend();

plt.ioff();
plt.style.use('fivethirtyeight')

ng = data[data.price <500]

plt.figure(figsize=(10,6))

sns.boxplot(y="price",x ='neighbourhood_group' ,data = ng)

plt.title("neighbourhood_group price distribution < 500")

plt.show()
def rank_price(hotel_price):

    if hotel_price<=75:

        return 'Low'

    elif hotel_price >75 and hotel_price<=500:

        return 'Medium'

    else:

        return 'High'

    
data['price'].apply(rank_price).value_counts().plot(kind='bar');
print ("Total Neighbourhoods: ", len(np.unique(data.neighbourhood)))
#using groupby to group two columns neighbourhood and price also find mean of price along with sorting the values and resetting index

df_top_prices_by_neighbourhood = data.groupby('neighbourhood').agg({'price': 'mean'}).sort_values('price').reset_index()
df_top_prices_by_neighbourhood.columns
plt.figure(figsize=(12,6))

sns.barplot(y="neighbourhood", x="price", data=df_top_prices_by_neighbourhood.head(10))

plt.ioff()
#method2

plt.figure(figsize=(12,6))

sns.barplot(y="neighbourhood", x="price", data=data.nlargest(10,['price']))

plt.ioff()
plt.style.use('fivethirtyeight')

f,ax=plt.subplots(1,2,figsize=(18,8))

data['room_type'].value_counts().plot.pie(explode=[0,0.05,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Share of Room Type')

ax[0].set_ylabel('Room Type Share')

sns.countplot('room_type',data=data,ax=ax[1],order=data['room_type'].value_counts().index)

ax[1].set_title('Share of Room Type')

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

roomdf = data.groupby('room_type').size()/data['room_type'].count()*100

labels = roomdf.index

values = roomdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
plt.figure(figsize=(10,6))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = data)

plt.title("Room types occupied by the neighbourhood_group")

plt.show()
# soure: previous project

plt.figure(figsize=(10,6))

sns.set_style("whitegrid");

sns.FacetGrid(data,hue='room_type',size=5).map(plt.scatter,'price','minimum_nights').add_legend()

plt.ioff()

plt.show()
#catplot room type and price

plt.figure(figsize=(10,6))

sns.catplot(x="room_type", y="price", data=data);

plt.ioff()
plt.style.use('fivethirtyeight')

fig,ax=plt.subplots(1,2,figsize=(15,8))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

data.neighbourhood.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])

ax[0].set_title("Top 10 neighbourhood by the number of rooms",size=20)

ax[0].set_xlabel('rooms',size=18)





count=data['neighbourhood'].value_counts()

groups=list(data['neighbourhood'].value_counts().index)[:10]

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 

plt.subplots_adjust(wspace =0.5, hspace =0)

plt.ioff()

plt.ylabel('')
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in data.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="yellow").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#let's comeback now to the 'name' column as it will require litte bit more coding and continue to analyze it!



#initializing empty list where we are going to put our name strings

_names_=[]

#getting name strings from the column and appending it to the list

for name in data.name:

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

_top_25_w=Counter(_names_for_count_).most_common()

_top_25_w=_top_25_w[0:25]
#now let's put our findings in dataframe for further visualizations

sub_w=pd.DataFrame(_top_25_w)

sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)
#we are going to use barplot for this visualization

plt.figure(figsize=(10,6))

viz_5=sns.barplot(x='Words', y='Count', data=sub_w)

viz_5.set_title('Counts of the top 25 used words for listing names')

viz_5.set_ylabel('Count of words')

viz_5.set_xlabel('Words')

viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80);
plt.figure(figsize=(10,6))

data['number_of_reviews'].plot(kind='hist')

plt.xlabel("Price")

plt.ioff()

plt.show()
df1=data.sort_values(by=['number_of_reviews'],ascending=False).head(1000)

df1.head()
import folium

from folium.plugins import MarkerCluster

from folium import plugins

print('Rooms with the most number of reviews')

Long=-73.80

Lat=40.80

mapdf1=folium.Map([Lat,Long],zoom_start=10,)



mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon,label in zip(df1.latitude,df1.longitude,df1.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1
sns.distplot(data[(data['minimum_nights'] <= 30) & (data['minimum_nights'] > 0)]['minimum_nights'], bins=31)

plt.ioff()
plt.figure(figsize=(10,6))

plt.scatter(data.longitude, data.latitude, c=data.availability_365, cmap='spring', edgecolor='black', linewidth=1, alpha=0.75)



cbar = plt.colorbar()

cbar.set_label('availability_365')
# Preparing the data 

data.drop(['name','id','host_name','last_review'],axis=1,inplace=True)

data['reviews_per_month']=data['reviews_per_month'].replace(np.nan, 0)
'''Encode labels with value between 0 and n_classes-1.'''

le = preprocessing.LabelEncoder()                                            # Fit label encoder

le.fit(data['neighbourhood_group'])

data['neighbourhood_group']=le.transform(data['neighbourhood_group'])    # Transform labels to normalized encoding.



le = preprocessing.LabelEncoder()

le.fit(data['neighbourhood'])

data['neighbourhood']=le.transform(data['neighbourhood'])



le = preprocessing.LabelEncoder()

le.fit(data['room_type'])

data['room_type']=le.transform(data['room_type'])



data.sort_values(by='price',ascending=True,inplace=True)



data.head()
'''Reversing Labeling Transform'''

#list(le.inverse_transform(data['room_type']))[:10]
'''Train LRM'''

lm = LinearRegression()



X = data[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]

y = data['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



lm.fit(X_train,y_train)
'''Get Predictions & Print Metrics'''

predicts = lm.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, predicts)),

        r2_score(y_test,predicts) * 100,

        mean_absolute_error(y_test,predicts)

        ))
error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': predicts.flatten()}).head(20)



error_airbnb.head(5)
title=['Pred vs Actual']

fig = go.Figure(data=[

    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),

    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])

])



fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(16,8))

sns.regplot(predicts,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.grid(False)

plt.show()
'''Gradient Boosted Regressor'''

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)

GBoost.fit(X_train,y_train)
'''Get Predictions & Metrics'''

predicts2 = GBoost.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, predicts2)),

        r2_score(y_test,predicts2) * 100,

        mean_absolute_error(y_test,predicts2)

        ))
error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': predicts2.flatten()}).head(20)



error_airbnb.head(5)
title=['Pred vs Actual']

fig = go.Figure(data=[

    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),

    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])

])



fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(16,8))

sns.regplot(predicts2,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Gradient Boosted Regressor model Predictions")

plt.show()