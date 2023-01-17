#importing required dependency



import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

from PIL import Image

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.tools as tls

import warnings

from collections import Counter

import squarify

from scipy.stats import skew, boxcox, norm, probplot





warnings.filterwarnings('ignore')
working_dir= "../input/new-york-city-airbnb-open-data/"



#load the data in pandas dataframe



df= pd.read_csv(working_dir+ "AB_NYC_2019.csv")

df.head()
rows = df.shape[0]

columns = df.shape[1]

print("The train dataset contains {0} rows and {1} columns".format(rows, columns))



Counter(df.dtypes.values)
df.info()
df_info= pd.DataFrame({"Dtype": df.dtypes, "Unique": df.nunique(), "Missing%": (df.isnull().sum()/df.shape[0])*100})

df_info
from scipy import stats



sns.distplot(df['price'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df['price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df['price'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column



sns.distplot(np.log1p(df["price"]) , fit=norm);



(mu, sigma) = norm.fit(np.log1p(df["price"]))

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(np.log1p(df["price"]), plot=plt)

plt.show()
colormap = plt.cm.magma



fig= plt.figure(figsize= (10,5))

sns.boxplot(df.neighbourhood_group, np.log1p(df["price"]))

plt.xlabel("log(Price)", size= 12)

plt.title("Borough wise distribution of log(price)", size= 20)

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(df[df.minimum_nights<30].minimum_nights)

plt.title('Minimum no. of nights distribution')

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(df.availability_365, color= "blue")

plt.title('Distribution of availability in days')

plt.show()
price_df= df.groupby("neighbourhood")['price'].mean().reset_index()



trace = go.Scatter(

    y = price_df.price,

    x = price_df.neighbourhood,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = price_df.price*.1,

        #color = np.random.randn(500), #set color equal to a variable

        color = price_df.price,

        colorscale='Portland',

        showscale=True

    )

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Mean Price by Regions',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Mean Price',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
f,ax=plt.subplots(1,2,figsize=(15,5))

df['room_type'].value_counts().plot.pie(explode=[0,0.05,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Share of Room Type',size= 16)

sns.countplot('room_type',data=df,ax=ax[1],order=df['room_type'].value_counts().index)

ax[1].set_title('Share of Room Type', size= 16)

plt.show()
f,ax=plt.subplots(1,2,figsize=(16,6))

df['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.05,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Share of Neighbourhood group', size= 18)

sns.countplot('neighbourhood_group',data=df,ax=ax[1],order=df['neighbourhood_group'].value_counts().index)

ax[1].set_title('Share of Neighbourhood group',size=18)

plt.show()


data = [go.Bar(

            x = df[df.room_type=="Entire home/apt"].neighbourhood.unique()[:50],

            y = df[df.room_type=="Entire home/apt"].neighbourhood.value_counts().values[:50],

            marker= dict(colorscale='Jet',

                         color = df.neighbourhood.value_counts().values[:50]

                        ),

            text='Histogram of listing by region'

    )]



layout = go.Layout(

    title='Borough listings of Entire home'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
data = [go.Bar(

            x = df[df.room_type=="Private room"].neighbourhood.unique()[:50],

            y = df[df.room_type=="Private room"].neighbourhood.value_counts().values[:50],

            marker= dict(colorscale='Viridis',

                         color = df.neighbourhood.value_counts().values[:50]

                        ),

            text='Histogram of listing by region'

    )]



layout = go.Layout(

    title='Borough listings of Private rooms'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
data = [go.Bar(

            x = df[df.room_type=="Shared room"].neighbourhood.unique()[:50],

            y = df[df.room_type=="Shared room"].neighbourhood.value_counts().values[:50],

            marker= dict(colorscale='Portland',

                         color = df.neighbourhood.value_counts().values[:50]

                        ),

            text='Histogram of listing by region'

    )]



layout = go.Layout(

    title='Borrow listings of shared room'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
from wordcloud import WordCloud, STOPWORDS



text= df['name'].values



plt.figure(figsize=(16,10))

wc = WordCloud(background_color="black", max_words=1000, stopwords=STOPWORDS, max_font_size= 40)

wc.generate(" ".join([str(x) for x in text]))

plt.title("Wordcloud of descriptions on Airbnb", fontsize=20)

# plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=1.0)

plt.axis('off')
plt.figure(figsize=(12,8))

sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)

plt.ioff()
import folium

from folium.plugins import HeatMap



m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'green',0.4:'yellow',0.6:'orange',1.0:'red'}).add_to(m)

display(m)