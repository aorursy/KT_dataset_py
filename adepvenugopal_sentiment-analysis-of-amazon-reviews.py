#Data processing packages

import pandas as pd

import numpy as np

pd.set_option('display.max_colwidth', 200)



#Visualization packages

import matplotlib.pyplot as plt

import seaborn as sns



#NLP packages

from textblob import TextBlob



import warnings

warnings.filterwarnings("ignore")
#Testing NLP - Sentiment Analysis using TextBlob

TextBlob("The movie is good").sentiment
#Importing YouTube comments data

data = pd.read_csv('../input/Reviews.csv',encoding='utf8',error_bad_lines=False);#opening the file UScomments
from wordcloud import WordCloud



def wc(data,bgcolor,title):

    plt.figure(figsize = (50,50))

    wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 50)

    wc.generate(' '.join(data))

    plt.imshow(wc)

    plt.axis('off')
#Displaying the first 5 rows of the data

data.head()
#Finding the size of the data

data.shape
#Extracting 1000 random samples from the data

comm = data.sample(2000)

comm.shape
#Calculating the Sentiment Polarity and Subjectivity

polarity=[] # list which will contain the polarity of the comments

subjectivity=[] # list which will contain the subjectivity of the comments

for i in comm['Text'].values:

    try:

        analysis =TextBlob(i)

        polarity.append(analysis.sentiment.polarity)

        subjectivity.append(analysis.sentiment.subjectivity)

        

    except:

        polarity.append(0)

        subjectivity.append(0)
#Adding the Sentiment Polarity column to the data

comm['polarity']=polarity

comm['subjectivity']=subjectivity
comm.polarity.hist(bins=50)
comm.subjectivity.hist(bins=50)
#Displaying the POSITIVE comments

df_positive = comm[comm.polarity==1]

df_positive[['ProfileName','Summary','Text']].head(10)
wc(comm['Text'][comm.polarity>0.20],'black','Common Words' )
#Displaying the NEGATIVE comments

df_positive = comm[comm.polarity==-1]

df_positive[['ProfileName','Summary','Text']].head(10)
wc(comm['Text'][comm.polarity<-0.2],'black','Common Words' )
#Displaying the NEUTRAL comments

df_positive = comm[comm.pol==0]

df_positive[['ProfileName','Summary','Text']].head(10)
comm.pol.value_counts().plot.bar()

comm.pol.value_counts()
#Converting the polarity values from continuous to categorical

comm['pol'][comm.pol==0]= 0

comm['pol'][comm.pol > 0]= 1

comm['pol'][comm.pol < 0]= -1