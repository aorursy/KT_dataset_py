import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from datetime import datetime

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image # converting images into arrays

import os, sys

from bs4 import BeautifulSoup

import requests

import re #regular expressions, a powerful ways of defining patterns to match strings

#OCULTO

#EXPLORACION DEL DIRECTORIO



#1.Ver el actual directorio de trabajo

#os.getcwd()



#2.Ver que hay en la carpeta de trabajo

#os.listdir('/kaggle/input/venezuelainterinpresidenttweets')
df = pd.read_csv('../input/venezuelainterinpresidenttweets/guaidotuits.csv', sep=';',error_bad_lines=False)

# sorting data frame by date

df=df.sort_values("date", axis = 0, ascending = True)

df
#Let's see how many columns and rows are in the Dataset

df.shape
#Now let's see information about each of the columns, if there are boxes without values and the type of variable

df.info()
# 1. columns we don´t need

df=df.drop(['username','geo','mentions','hashtags'], axis=1)

### OCULTO ###  

#2. Fixing date column

#df['date']=df.date.str.split()

#df['Hour']=df.date.str.get(1)

#df['date']=df['date'].str.get(0) 

#df.head()
#2. change the type of object from str to datetime

# so we can use later to plot

df['date'] =  pd.to_datetime(df['date'])

                              #format='%y/%m/%d %M/%S')



#Set Date column as index column

df.set_index('date',inplace=True)







# Confirm the date column is in datetime format

df.info()
# 3. convert Id to int64 

#df["id"]= df["id"].astype(int) Doesn't work

#df['id'] = df['id'].apply(lambda x: x if pd.isnull(x) else str(int(x))) Doesn't work

df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(-1).astype(int)



df.info()
df.head()
#Let's take a closer look at the retweets column

#df.retweets.plot('hist')

#df['retweets'].plot('hist')

sns.distplot(df['retweets'])
# Let's see which are the tweets with more RTs of Guaidó

df[df.retweets > 50000]
#Let's take a closer look at the favorites column

#df.favorites.plot('hist')

sns.distplot(df['favorites'])
# Let's see which are the tweets with more Favorites of Guaidó user

df[df.favorites > 90000]

def plotguaido(df,figsize=(20,4)):

    # Set the width and height of the figure

    plt.figure(figsize=(20,4))



    # Add title

    plt.title("Popularity of Guaidó's tweets")



    # Line chart showing Retweets

    sns.lineplot(data=df['retweets'], label="Retweets")



    # Line chart showing Favorites

    sns.lineplot(data=df['favorites'], label="Favorites")



    # Add label for horizontal axis

    plt.xlabel("Date")

    plt.ylabel("Number of RTs and ♥")

plotguaido(df)
# First let's see the a wordcloud of all tweets of Juan Guaidó

def wordcloudguaido(df):

    comment_words = ' '

    stopwords=(['el','la','es','son','a','en','lo','cualquier','http','pic','www','twitter',

                'para','ante','por','no','nos','con','que','con','com','hasta','bajo','de',

                'su','los','como','del','hoy','se','las','una','más','éste','este','esté',

                'mi','ha','hemos','https','un','unos','desde','todo','asambleave','han','ni'

                ,'al','hay','ya','me','statu','donde','sin','sobre','tras','está','ésta','están'

                'éstas','toda','todas','le','cada','su','sus','pero','nuestra','pscp','esta',

                'les','como','quien','sido','aún','eso','tv','será','vez','solo','van','están',

                'porque','qué','tienen','aquí','sta','tus','ahora','esa','ello','cómo','sea',

                'ser','va','tiene','tatus','tatu','si','ver','uno','fue','gran','traves',

                'través','vez','st','atus','st atus','stat us','entre','hacia','hacía','23f','asi',

                'esto','ésto','ellos','ellas','tanto','tantas','sigue','sigues','siguen','mucho',

                'hace','nunca','tener','cuando','hacer','status','somos','nuestra','nuestro',

                'así','ese','ése','pueden','pesar','debe','hecho','necesaria','nuestras',

                'europaestáconvenezuela','quienes','ustedes','nuestros','23favalanchahumanitaria'])

    #stopwords = set(STOPWORDS) 



    # iterate through the csv file 

    for val in df.text: 



        # typecaste each val to string 

        val = str(val) 



        # split the value 

        tokens = val.split() 



        # Converts each token into lowercase 

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 



        for words in tokens: 

            comment_words = comment_words + words + ' '





    wordcloud = WordCloud(width = 800, height = 800, 

                    background_color ='white', 

                    stopwords = stopwords, 

                    min_font_size = 10).generate(comment_words) 



    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show() 

wordcloudguaido(df)
#OCULTO

#Let´s put all these words in a President Guaido Mask

#1. create a mask file importing the .png image 

img = np.array(Image.open('../input/venezuelainterinpresidenttweets/guaidomask.png'))



#OCULTO

sns.set(rc={'figure.figsize':(11.7,8.27)})



wordcloud = WordCloud(mask=img,background_color="white",stopwords=stopwords).generate(comment_words)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

#plt.title('Responsibilites',size=24)

plt.show()
#1st. Trimester Dataframe

#create a new dataframe. Date between January and March

df1=df.loc['2019-01-01':'2019-04-01'].sort_values('date',axis=0,ascending = True)

df1.head()
plotguaido(df1)
#Let's see closer

#PEAK 1

dfpeak1=df1.loc['2019-01-15':'2019-02-15'].sort_values('retweets',axis=0,ascending = False)#.drop(['permalink'],axis=1)

dfpeak1.head(20)
plotguaido(dfpeak1)
#What did Guaidó's tweets say during these days?

wordcloudguaido(dfpeak1)
#2nd PEAK

dfpeak2=df1.loc['2019-02-15':'2019-03-15'].sort_values('retweets',axis=0,ascending = False).drop(['permalink'],axis=1)

dfpeak2.head(20)
plotguaido(dfpeak2)
#What did Guaidó's tweets say during these days?

wordcloudguaido(dfpeak2)
#1. Create a new dataset that contains replies of most important tweet during every period

#2. Create a replies wordcloud 

#3. Create a sentiment Analysis of the replies

#4. Plot the result 
# FIRST TRIMESTER 

    # Peak 1

        # Let's select more relevants from this period

dfpeak1.head(1)

#In Process

def replyfinder(url=''):

    url = 'https://twitter.com/jguaido/status/1088261421023088640'

    source = requests.get(url).text

    soup = BeautifulSoup(source,'html.parser')



    search=soup.find_all(class_="TweetTextSize") #búsqueda de líneas con dicho 'str' dentro del class



    dfreplies=pd.DataFrame() #hace un Dataset vacío

    dfreplies['Replies']=search #inserta la búsqueda en el Dataframe

    dfreplies




url = 'https://twitter.com/jguaido/status/1088261421023088640'

source = requests.get(url).text

soup = BeautifulSoup(source,'html.parser')



search=soup.find_all(class_="TweetTextSize") #búsqueda de líneas con dicho 'str' dentro del class

dfreplies=pd.DataFrame() #hace un Dataset vacío

dfreplies['Replies']=search #inserta la búsqueda en el Dataframe

dfreplies
#Create dataframe with sample

#dfpk1sample=dfpeak1.sample(10, replace=True)

#dfpk1sample
#Mar=df.loc['2019-02-01':'2019-03-01'].sort_values('retweets',axis=0,ascending = False)

#Mar=Mar.shape[0]

#March



#Jandf=df.loc['2019-01-01':'2019-02-01'].sort_values('retweets',axis=0,ascending = False).head(63)

#Febdf=df.loc['2019-02-01':'2019-03-01'].sort_values('retweets',axis=0,ascending = False).head(65)

#Mardf=df.loc['2019-03-01':'2019-04-01'].sort_values('retweets',axis=0,ascending = False).head(65)

#Aprdf=df.loc['2019-04-01':'2019-05-01'].sort_values('retweets',axis=0,ascending = False).head(65)

#Maydf=df.loc['2019-05-01':'2019-06-01'].sort_values('retweets',axis=0,ascending = False).head(65)

#Jundf=df.loc['2019-06-01':'2019-07-01'].sort_values('retweets',axis=0,ascending = False).head(65)

#Juldf=df.loc['2019-07-01':'2019-08-01'].sort_values('retweets',axis=0,ascending = False).head(65)



#Mardf.loc['2019-03-01':'2019-04-01'].sort_values('retweets',axis=0,ascending = False)
#df=df.sort_values("retweets", axis = 0, ascending = False)

#df