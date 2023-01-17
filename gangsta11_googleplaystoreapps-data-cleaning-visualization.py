import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected = True)

import plotly.graph_objs as go
import os

print(os.listdir("../input/google-play-store-apps"))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df =pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.head()
df.shape
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(6)


df.dropna(how ='any', inplace = True)
df.shape
df.isnull().sum().head()
df.dtypes.value_counts()
df['Size'].value_counts()
#A function to convert  Kb to Mb

def convert_size(size):

    if 'M' in size:

        x = size[:-1]

        x = float(x)

        return(x)

    elif 'k' in size:

        x = size[:-1]

        x = float(x)/1000

        return(x)

    else:

        return None



df['Size'] = df['Size'].map(convert_size)



#filling Size which had NA (Rows with Size == 'Varies with device' )



df.Size.fillna(method = 'ffill', inplace = True)



#('ffill method')propagates last valid observation forward to next valid
df.sort_values(by = 'Size').head(1)
df['Installs'].value_counts()
#Removing ',' and '+' from 'Installs' column

df["Installs"] = df["Installs"].str.replace("+","")

df["Installs"] = df["Installs"].str.replace(",","")

df["Installs"] = pd.to_numeric(df["Installs"])
df.head(1)
df['Reviews'].value_counts()
df["Reviews"] = pd.to_numeric(df["Reviews"])
df['Type'].value_counts()
#Coverting 'Type' column into binary values as ('Free' == 0) and ('Paid' == 1)

def converttype(cost):

    if 'Free' in cost:

        return 0

    elif 'Paid' in cost:

        return 1

    else:

        return None

    

df['Type'] = df['Type'].map(converttype)
df['Type'].value_counts()
df['Price'].value_counts()
#Removing '$' from 'Price' column

df["Price"] = df["Price"].str.replace("$","")

df["Price"] = pd.to_numeric(df["Price"])
df.head(1)
df['Last Updated'].head()
from datetime import datetime,date

dt=pd.to_datetime(df['Last Updated'])

dt.head()
df['Last Updated Days'] = dt.apply(lambda x:date.today()-datetime.date(x))

df['Last Updated Days'].head()
df.drop(['Last Updated'],axis=1,inplace=True)
df.head(1)
df['Genres'].unique()
sep = ';'

primary = df['Genres'].apply(lambda x: x.split(sep)[0])

df['Primary Genre']=primary

df['Primary Genre'].head()
primary = df['Genres'].apply(lambda x: x.split(sep)[-1])

primary.unique()

df['Secondary Genre']= primary

df['Secondary Genre'].head()
grouped = df.groupby(['Primary Genre','Secondary Genre'])

grouped.size().head()
df.drop(['Genres','Current Ver','Android Ver'],axis=1,inplace=True)
df.head(1)
fig = plt.figure(figsize = (5,5))

labels = ['Free', 'Paid'] 

size = df['Type'].value_counts()

#colors = plt.cm.Wistia(np.linspace(0, 1, 5))

colors = ['r','k','y']

explode = [0, 0.1]



plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Free Apps vs Paid Apps', fontsize = 15)

plt.legend()

plt.show()
fig = plt.figure(figsize = (8,6))

sns.countplot(x = 'Content Rating' , data = df)
fig = plt.figure(figsize = (10,5))

sns.countplot(x='Type' , hue = 'Content Rating' , data = df)

plt.title('Type vs Content Rating', fontsize = 15)

plt.show()
fig = plt.figure(figsize = (15,15))

pos = df.groupby(by='Category').size().reset_index()

pos.columns = ['Category','Count']



labels = pos['Category']

values = pos['Count']

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label', textinfo='percent', textfont_size=10,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Apps based on Category")

fig.show()
MostInstalled = df.sort_values(by = 'Installs' , ascending = False).head(20)

BestRating = MostInstalled.sort_values(by = 'Rating' , ascending = False).head(10)
fig = plt.figure(figsize = (15,7))

sns.barplot(x='App',y='Rating' , data = BestRating)
MaxReviews = df.sort_values(by = 'Reviews',ascending = False).head(10)
fig = plt.figure(figsize = (13,5))

sns.barplot(x='App',y='Reviews' , data = MaxReviews)
sns.relplot(x="Rating", y="Reviews", hue = 'Type' , kind = 'line' , data=df)
sns.relplot(x="Rating", y="Installs", hue = 'Type' , kind = 'line' , data=df)
MaxSize = df.sort_values(by = 'Size' , ascending = False).head(1000)

BestRatingSize = MaxSize.sort_values(by = 'Rating').head()
fig = plt.figure(figsize = (14,6))

sns.barplot(x='App',y='Size',hue = 'Rating' , data = BestRatingSize)
GenreTable = pd.crosstab(index=df["Primary Genre"],columns=df["Secondary Genre"])

GenreTable.head()
GenreTable.plot(kind="barh", figsize=(15,15),stacked=True);

plt.legend(bbox_to_anchor=(1.0,1.0))
days = df.sort_values(by = 'Last Updated Days' , ascending = False).head(5)
g = sns.catplot(x="Installs", y="Last Updated Days",hue="Rating", col="App",data=days, kind="bar",height=4, aspect=.7)

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Outdated Apps')
Social = df[(df['Category'] == 'SOCIAL') | (df['Category'] == 'GAME')]

SocialMaxReviews = Social.sort_values(by = 'Reviews' , ascending = False).head(10)
g = sns.FacetGrid(data = SocialMaxReviews, row = 'Category' , col  = 'App',hue = 'Content Rating' , margin_titles = True)

g = (g.map(plt.scatter , "Size" , "Reviews"))

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Top "Social" and "Game" Category Apps based on their Reviews')
PaidApp = df[df['Type'] == 1]

CostlyApp = PaidApp.sort_values(by = 'Price' , ascending = False).head(50)

CostlyAppWorstRating = CostlyApp.sort_values(by = 'Rating').head()
g = sns.FacetGrid(data = CostlyAppWorstRating, col  = 'App')

g = (g.map(plt.scatter , "Rating" , "Price"))

plt.subplots_adjust(top=0.8)

g.fig.suptitle('High priced Apps with worst Rating')
g = sns.PairGrid(df , hue = 'Content Rating')

g.map_diag(plt.hist,edgecolor = 'k')

g.map_offdiag(plt.scatter,edgecolor = 'k')

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Special Attributes distribution of Apps based on their Content Rating')