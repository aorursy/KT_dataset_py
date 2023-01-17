



import numpy as np 

import pandas as pd 

import re

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.graph_objects as go

!pip install pywaffle

!pip install --upgrade pip

!pip install squarify

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df.Installs = df.Installs.str.replace('+' , '')

df.Installs = df.Installs.str.replace(',' , '')

df.dropna(inplace = True) 

 





#dropping duplicates

df = df.drop_duplicates(subset='App', keep="first")



df.Installs = df.Installs.astype('float').astype('int')

df.head(5)

df1 = df.groupby(['Category'])['Installs'].sum().sort_values(ascending = False).reset_index()





from pywaffle import Waffle

sns.set_context("notebook")



fig = plt.figure(

    FigureClass=Waffle, 

    rows=10, 

    columns = 24,

    figsize = (20, 10),

    values=df1.Installs/1000000000, 

    labels= list(df1.Category),

    legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)},

    starting_location='NW'

    

)

fig.set_facecolor('#EEEEEE')

#plt.title('Installations Distribution of Categories', size = 20)

plt.show()

 
dfp = df.groupby(['Category'])['Installs'].sum().sort_values(ascending = False).reset_index()





category = list(dfp.Category)

installs = list(dfp.Installs)



fig= go.Figure(go.Treemap(

    

    labels =  category,

    parents=[""]*len(category),

    values =  installs,

    textinfo = "label+percent entry"

))

fig.update_layout(

    autosize=False,

    width= 800,

    height=800,)



fig.show()
df1.Installs = df1.Installs/1000000000# converting into billions

df2 = df1.head(15)

plt.figure(figsize = (14,10))

sns.set_context("talk")

sns.set_style("darkgrid")





ax = sns.barplot(x = 'Installs' , y = 'Category' , data = df2 )

ax.set_xlabel('No. of Installations in Billions')

ax.set_ylabel('')

ax.set_title("Most Popular Categories in Play Store", size = 20)

df3 = df1.tail(15)

sns.set_context("talk")



plt.figure(figsize = (14,10))

ax = sns.barplot(x = 'Installs' , y = 'Category' , data = df3 )

ax.set_xlabel('No. of Installations in Billions')

ax.set_ylabel('')

ax.set_title("Least Popular Categories in Play Store", size = 20)

dfa = df.groupby(['Category' ,'App'])['Installs'].sum().reset_index()



dfa = dfa.sort_values('Installs', ascending = False)



dftop = dfa[dfa.Installs == 1000000000 ]





dftop.style.set_properties(**{'background-color': 'black',

                            'color': 'lawngreen',

                            'border-color': 'white'})
size = df.groupby(['Category','App'])['Size'].sum().sort_values(ascending = False).reset_index()



# #dropping varies with device

size = size[size.Size!='Varies with device']

#size['new'] = size['Size'].str.extract('(\w)', expand=True) 





size[['n', 'k']] = size.Size.str.extract('([^a-zA-Z]+)([a-zA-Z]+)', expand=True) #extracting numbers and data size 

size['k'] = size['k'].map({'M': 1000, 'k': 1}) #renmaing k and m 

size.Size = size.Size.str.replace('+' , '')

size.Size = size.Size.str.replace(',' , '')

size.dropna(inplace = True) 

size.n = size.n.astype('float')

size['truesize'] = size.n * size.k



size = size[['Category','App', 'truesize']]   #truesize is in KB



size = size.sort_values('truesize', ascending = False)



largesize = size[size.truesize == 100000.0]



largesize.style.set_properties(**{'background-color': 'black',

                            'color': 'lawngreen',

                            'border-color': 'white'})
largesize.App.unique()
sns.set_context("talk")



plt.figure(figsize = (14,10))



ax = sns.distplot(size['truesize'])

ax.set_xlabel('App size in KB')

ax.set_title('App Size Distribution')

plt.figure(figsize=(13,10), dpi= 80)

sns.set_context("notebook")

ax = sns.boxplot(x='truesize', y='Category', data= size , notch=False)

ax.set_title('App Size Distribution ' , size = 20)
apps = ['GAME', 'COMMUNICATION', 'TOOLS', 'PRODUCTIVITY', 'SOCIAL',

       'PHOTOGRAPHY','FAMILY', 'VIDEO_PLAYERS' ]
dfa.Installs = dfa.Installs/1000000
sns.set_context("poster")

sns.set_style("darkgrid")



plt.figure(figsize=(40,30))



for i,app in enumerate(apps):

    df2 = dfa[dfa.Category == app]

    df3 = df2.head(5)

    plt.subplot(4,2,i+1)

    sns.barplot(data= df3,x= 'Installs' ,y='App' )

    plt.xlabel('Installation in Millions')

    plt.ylabel('')

    plt.title(app,size = 20)

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()


columns = [ 'TRAVEL_AND_LOCAL','NEWS_AND_MAGAZINES','ENTERTAINMENT', 'BOOKS_AND_REFERENCE',

            'PERSONALIZATION', 'SHOPPING', 'HEALTH_AND_FITNESS', 'SPORTS']



sns.set_context("poster")

sns.set_style("darkgrid")



plt.figure(figsize=(40,30))



for i,column in enumerate(columns):

    df2 = dfa[dfa.Category == column]

    df3 = df2.head(5)

    plt.subplot(4,2,i+1)

    sns.barplot(data= df3,x= 'Installs' ,y='App' )

    plt.xlabel('Installation in Millions')

    plt.ylabel('')

    plt.title(column,size = 20)

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()


mapps = ['BUSINESS', 'LIFESTYLE', 'MAPS_AND_NAVIGATION', 'FINANCE','WEATHER', 'EDUCATION', 'FOOD_AND_DRINK', 'DATING']



sns.set_context("poster")

sns.set_style("darkgrid")



plt.figure(figsize=(40,30))



for i,mapp in enumerate(mapps):

    df2 = dfa[dfa.Category == mapp]

    df3 = df2.head(5)

    plt.subplot(4,2,i+1)

    sns.barplot(data= df3,x= 'Installs' ,y='App' )

    plt.xlabel('Installation in Millions')

    plt.ylabel('')

    plt.title(mapp,size = 20)

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()


sapps = [ 'ART_AND_DESIGN', 'HOUSE_AND_HOME', 'AUTO_AND_VEHICLES',

        'COMICS', 'MEDICAL', 'PARENTING', 'BEAUTY',

       'EVENTS']



sns.set_context("paper")

sns.set_style("darkgrid")



plt.figure(figsize=(30,30))



for i,sapp in enumerate(sapps):

    df2 = dfa[dfa.Category == sapp]

    df3 = df2.head(5)

    plt.subplot(4,2,i+1)

    ax =sns.barplot(data= df3,x= 'Installs' ,y='App' )

    plt.xlabel('Installation in Millions')

    plt.ylabel('')

    plt.title(sapp,size = 20)

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()
rating = df.groupby(['Category','Installs', 'App'])['Rating'].sum().sort_values(ascending = False).reset_index()



toprating = rating[rating.Rating == 5.0]



toprating
plt.figure(figsize=(13,10), dpi= 80)

sns.set_context("notebook")

ax = sns.boxplot(x='Rating', y='Category', data= rating , notch=False)



ax.set_title('Distribution of App Ratings' , size = 20)
