#Data processing packages

import pandas as pd

import numpy as np

pd.set_option('display.max_colwidth', 300)



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

data = pd.read_csv('../input/amazon_alexa.tsv', delimiter='\t')
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
comm = data

comm.shape
#Calculating the Sentiment Polarity

polarity=[] # list which will contain the polarity of the comments

subjectivity=[] # list which will contain the subjectivity of the comments

for i in comm['verified_reviews'].values:

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
comm[comm.polarity<0].head(10)
#Displaying the POSITIVE comments

comm[comm.polarity>0].head(10)
#Displaying the NEGATIVE comments

comm[comm.polarity<0].head(10)
#Displaying the NEUTRAL comments

comm[comm.polarity==0].head(10)
#Displaying highly subjective reviews

comm[comm.subjectivity>0.8].head(10)
#Displaying highly positive reviews

comm[comm.polarity>0.75].head(10)
wc(comm['verified_reviews'][comm.polarity>0.75],'black','Common Words' )
#Displaying highly negative reviews

comm[comm.polarity<-0.25].head(10)
wc(comm['verified_reviews'][comm.polarity<-0.25],'black','Common Words' )
comm.polarity.hist(bins=50)
comm.subjectivity.hist(bins=50)
#Converting the polarity values from continuous to categorical

comm['polarity'][comm.polarity==0]= 0

comm['polarity'][comm.polarity > 0]= 1

comm['polarity'][comm.polarity < 0]= -1
comm.polarity.value_counts().plot.bar()

comm.polarity.value_counts()