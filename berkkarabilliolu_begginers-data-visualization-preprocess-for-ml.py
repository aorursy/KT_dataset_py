# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import bokeh

from bokeh.io import show, output_notebook

from bokeh.palettes import Spectral9

from bokeh.plotting import figure

output_notebook() # You can use output_file();



import plotly.graph_objects as go

import plotly.express as px



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.offline as py

py.init_notebook_mode(connected = True)



# Special

import wordcloud, missingno

from wordcloud import WordCloud # wordcloud

import missingno as msno # check missing value

import networkx as nx



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data.head()
data.describe()
data.shape
data.info()
#missing data

data.isnull().sum()
data.columns
import missingno as msno
# Visualise the missing ones

msno.matrix(data)
data.dropna(how = 'any', inplace = True)
data.isnull().sum()
g = sns.countplot(x="Category",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

plt.title('Category Counter',size = 20)
data = pd.get_dummies(data, columns = ['Category'])
data.drop(['Current Ver','Last Updated','Android Ver'], axis = 1, inplace = True)
labels =data['Type'].value_counts(sort = True).index

sizes = data['Type'].value_counts(sort = True)





colors = ["blue","red"]

explode = (0.1,0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('How much free apps?',size = 20)

plt.show()
data['Free'] = data['Type'].map(lambda s :1  if s =='Free' else 0)

data.drop(['Type'], axis=1, inplace=True)
#Cleaning the installs with replace "+ to ' '"

data.Installs = data.Installs.apply(lambda x: x.replace(',',''))

data.Installs = data.Installs.apply(lambda x: x.replace('+',''))

data.Installs = data.Installs.apply(lambda x: int(x))
sortedv = sorted(list(data['Installs'].unique()))
#Encoding

data['Installs'].replace(sortedv,range(0,len(sortedv),1), inplace = True )
data.Installs.value_counts()
#Scatter Plot Installs- Reviews

fig = px.scatter(data, y= "Installs", x ="Reviews")

py.iplot(fig, filename = "test")
plt.figure(figsize = (12,10))

sns.regplot(x="Installs", y="Rating", color = 'purple',data=data);

plt.title('Rating-Installs',size = 15)
hist = data.hist(figsize =(50,50))
data['Price'].value_counts().head(10)
#Cleaning the Price from $ to ' '

data.Price = data.Price.apply(lambda x: x.replace('$',''))

data['Price'] = data['Price'].apply(lambda x: float(x))
data['Price'].describe()
#We can give a number to string values

data.loc[data['Price'] == 0, 'priceint'] = '0'

data.loc[(data['Price'] > 0) & (data['Price'] <= 1), 'priceint'] = '1'

data.loc[(data['Price'] > 1) & (data['Price'] <= 3), 'priceint']   = '2'

data.loc[(data['Price'] > 3) & (data['Price'] <= 5), 'priceint']   = '3'

data.loc[(data['Price'] > 5) & (data['Price'] <= 15), 'priceint']   = '4'

data.loc[(data['Price'] > 15), 'priceint']  = '5'
#Rating-Priceint

data[['priceint', 'Rating']].groupby(['priceint'], as_index=False).mean()
g = sns.catplot(x="priceint",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Pastel1")

g.despine(left=True)

g.set_xticklabels(rotation=90)

g = g.set_ylabels("Rating")

plt.title('Catplot Rating VS Priceint',size = 10)

data['priceint'] = data['priceint'].astype(int)
g = sns.kdeplot(data.Rating, color = 'Purple', shade = True)

g.set_xlabel('Rating')

g.set_ylabel('Frequency')

plt.title('Rating Frequency',size = '25')
#We need to extract string values

data.Size.unique()
len(data[data.Size == 'Varies with device'])
#We replace Varies with device to Nan 

data['Size'].replace('Varies with device', np.nan, inplace = True )
#Some cleaning and string to float

data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \

             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)

            .fillna(1)

            .replace(['k','M'], [10**3, 10**6]).astype(int))
#Instead of dropping for too many missing values, we write values that do not change the mean standard deviation from the relation with Genres.

data['Size'].fillna(data.groupby('Genres')['Size'].transform('mean'),inplace = True)
# we have 1 unrated value 

data[data['Content Rating']=='Unrated']
data = data[data['Content Rating'] != 'Unrated']
#one hot encoding

data = pd.get_dummies(data, columns = ['Content Rating'])
data.Genres.value_counts()
data['Genres'] = data['Genres'].str.split(';').str[0]
data.Genres.value_counts()
# Music And Music & Audio Columns is the same Genres So we can apply it

data['Genres'].replace('Music & Audio', 'Music',inplace = True)
g = sns.countplot(x="Genres",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

plt.title('Count of app in each Genres',size = 20)
g = sns.catplot(x="priceint",y = "Rating",hue = "Genres" , kind = 'swarm' , data=data , palette = "Paired", size = 15)

g.set_xticklabels(rotation = 90)

plt.title('Genres in each Prices vs Ratings',size = 20)
data.loc[data['Rating'] < 0.5,'Ratingint'] = '0'

data.loc[(data['Rating'] >= 0.5) & (data['Rating'] < 1.5),'Ratingint'] = '1'

data.loc[(data['Rating'] >= 1.5) & (data['Rating'] < 2.5),'Ratingint'] = '2'

data.loc[(data['Rating'] >= 2.5) & (data['Rating'] < 3.5),'Ratingint'] = '3'

data.loc[(data['Rating'] >= 3.5) & (data['Rating'] < 4.5),'Ratingint'] = '4'

data.loc[(data['Rating'] >= 4.5) & (data['Rating'] < 5),'Ratingint'] = '5'
data.dropna(how = 'any', inplace = True)
data["Ratingint"]=data["Ratingint"].astype(int)
data.Ratingint.dtype
data.Rating.head(10)
data.Ratingint.head(10)
f ,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data[['Installs','Price','Size','Ratingint']].corr(), annot = True , ax=ax , cmap ="YlGnBu")

plt.show()
#and we need one hot encoding for ML

data = pd.get_dummies(data, columns = ['Genres'])
data.drop(['App','Reviews','Installs','Rating'],axis = 1 ,inplace = True)
data.corr()