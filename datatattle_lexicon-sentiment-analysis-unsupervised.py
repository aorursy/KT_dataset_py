# File Path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Loading libraries
import numpy as np # provides a high-performance multidimensional array and tools for its manipulation
import pandas as pd # for data munging, it contains manipulation tools designed to make data analysis fast and easy
import re # Regular Expressions - useful for extracting information from text 
import nltk # Natural Language Tool Kit for symbolic and statistical natural language processing
import spacy # processing and understanding large volumes of text
import string # String module contains some constants, utility function, and classes for string manipulation
import re

# For viz
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
pd.options.mode.chained_assignment = None
#Loading File
df = pd.read_csv('/kaggle/input/coronavirus-tweets/Corona_tweets.csv',encoding='latin1')
#Shape of dataframe
print(" Shape of training dataframe: ", df.shape)
# Drop duplicates
df.drop_duplicates()
print(" Shape of dataframe after dropping duplicates: ", df.shape)
#Null values

null= df.isnull().sum().sort_values(ascending=False)
total =df.shape[0]
percent_missing= (df.isnull().sum()/total).sort_values(ascending=False)

missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data= missing_data.rename(columns= { "index": " column name"})
 
print ("Null Values in each column:\n", missing_data)

!pip install vaderSentiment
import vaderSentiment
# calling SentimentIntensityAnalyzer object
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
# or use nltk


#import nltk
#nltk.download('vader_lexicon')
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

#analyser = SentimentIntensityAnalyzer()
# Using polarity scores for knowing the polarity of each text
def sentiment_analyzer_score(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
#testing the function
tweet  = "I would love to watch the magic show again"
tweet2 = "What the hell they have made. Pathetic!"
tweet3 = " I do not know what to do"  
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
tweet  = "I like the fact that monsoon is over"
tweet2 = "I LIKE the fact that monsoon is over"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
tweet  = "What is wrong with you"
tweet2  = "What is wrong with you?"
tweet3 = "What is wrong with you??"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
tweet  = "He is good but his mother is irritating"
tweet2 = "The thai curry was bad, however pasta was delicious"
tweet3 = "The thai curry was ok and pasta was delicious"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
tweet = "Real Madrid's game play was good last night."
tweet2 = "Real Madrid's game play was extremely good last night."
tweet3 = "Real Madrid's game play was somewhat good last night."
tweet4 = "Real Madrid's game play was terrible last night."
tweet5 = "Real Madrid's game play was awfully terrible last night."
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
print (sentiment_analyzer_score(tweet4))
print (sentiment_analyzer_score(tweet5))
tweet = " What a fine day I am having today"
tweet2 = " What a fine day I am having today :-)"
tweet3 = " What a fine day I am having today :-) :-)"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
tweet = "I love the team and how they played last night"
tweet2 = "I love the team and how they played last night 💘"
tweet3 = "I love the team and how they played last night 😁"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
tweet = "I am laughing like crazy"
tweet2 = "I am laughing like crazy lmao"
tweet3 = "I am laughing like crazy lol"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))
tweet = "He wasn't very good at the play"
tweet2 = "He was not very good at the play"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))

tweet = "He is kinda bored"
tweet2 = "He is friggin bored"
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df['text'].apply(tokenizer.tokenize)
words_descriptions.head()
all_words = [word for tokens in words_descriptions for word in tokens]
df['description_lengths']= [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
# Checking most common words
from collections import Counter
count_all_words = Counter(all_words)
count_all_words.most_common(100)
#### 1-gram tokenizer
example = 'The quick brown fox jumps over the lazy dog.'

# remove the dots and make all words lower case
clean_example = re.sub(r'\.', '', example)
print(clean_example.split())
# 2-gram tokenizer

example = 'The quick brown fox jumps over the lazy dog.'

without_first = example.split()[1:]
without_last = example.split()[:-1]

list(zip(without_last, without_first))
print (sentiment_analyzer_score(tweet2))
df['scores'] = df['text'].apply(lambda review: analyser.polarity_scores(review))

df.head()
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

df.head()
def Sentimnt(x):
    if x>= 0.05:
        return "Positive"
    elif x<= -0.05:
        return "Negative"
    else:
        return "Neutral"
#df['Sentiment'] = df['compound'].apply(lambda c: 'positive' if c >=0.00  else 'negative')
df['Sentiment'] = df['compound'].apply(Sentimnt)


df.head()
var1 = df.groupby('Sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
sns.set_style("white")
sns.set_palette("Set2")
var1.style.background_gradient()
plt.figure(figsize=(12,6))
sns.countplot(x='Sentiment',data=df)
fig = go.Figure(go.Funnelarea(
    text =var1.Sentiment,
    values = var1.text,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()
df['temp_list'] = df['text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
#temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()
# Tree of the most common words
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
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
comment_words = '' 
stopwords = set(STOPWORDS) 
  
df_positive = df[df["Sentiment"]== "Positive"] 
# iterate through the csv file 
for val in df_positive.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = "green") 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
comment_words = '' 
stopwords = set(STOPWORDS) 
  
df_negative = df[df["Sentiment"]== "Negative"] 
# iterate through the csv file 
for val in df_negative.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = "red") 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
comment_words = '' 
stopwords = set(STOPWORDS) 
  
df_neutral = df[df["Sentiment"]== "Neutral"] 
# iterate through the csv file 
for val in df_positive.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = "yellow") 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
del df_neutral
del df_positive
del df_negative
import warnings
warnings.filterwarnings('ignore')
from textblob import TextBlob, Word, Blobber
tweet = "I would love to watch the magic show again"
TextBlob(tweet).sentiment 
# Applying on dataset
df['TB_score']= df.text.apply(lambda x: TextBlob(x).sentiment)
df.head()
df['TB_sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment[0])
df.head()
!pip install nrclex
from nrclex import NRCLex
tweet = NRCLex('Good work to the team')
#Return affect dictionary
print(tweet.affect_dict)
#Return raw emotional counts
print("\n",tweet.raw_emotion_scores)
#Return highest emotions
print("\n", tweet.top_emotions)
#Return affect frequencies
print("\n",tweet.affect_frequencies)
text = NRCLex("Congratulations ")
# Getting top emotions
print("\n", text.top_emotions)
# Getting the top most emotion
print("\n", text.top_emotions[0][0])
# Getting the top most emotion score
print("\n", text.top_emotions[0][1])
text = NRCLex("We can do it ")
# Getting top emotions
print("\n", text.top_emotions)
# Getting the top most emotion
print("\n", text.top_emotions[0][0])
# Getting the top most emotion score
print("\n", text.top_emotions[0][1])
def emotion(x):
    text = NRCLex(x)
    if text.top_emotions[0][1] == 0.0:
        return "No emotion"
    else:
        return text.top_emotions[0][0]
df['Emotion'] = df['text'].apply(emotion)
df.head()

import matplotlib.pyplot as plt
from matplotlib import cm
from math import log10

df_chart = df[df.Emotion != "No emotion"]
labels = df_chart.Emotion.value_counts().index.tolist()
data = df_chart.Emotion.value_counts()
#number of data points
n = len(data)
#find max value for full ring
k = 10 ** int(log10(max(data)))
m = k * (1 + max(data) // k)

#radius of donut chart
r = 1.5
#calculate width of each ring
w = r / n 

#create colors along a chosen colormap
colors = [cm.terrain(i / n) for i in range(n)]

#create figure, axis
fig, ax = plt.subplots()
ax.axis("equal")

#create rings of donut chart
for i in range(n):
    #hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
    innerring, _ = ax.pie([m - data[i], data[i]], radius = r - i * w, startangle = 90, labels = ["", labels[i]], labeldistance = 1 - 1 / (1.5 * (n - i)), textprops = {"alpha": 0}, colors = ["white", colors[i]])
    plt.setp(innerring, width = w, edgecolor = "white")

plt.legend()
plt.show()
b = df_chart.Emotion.value_counts().index.tolist()
a = df_chart.Emotion.value_counts(normalize = True).tolist()
row = pd.DataFrame({'scenario' : []})
row["scenario"] = b
row["Percentage"] = a
fig = px.treemap(row, path= ["scenario"], values="Percentage",title='Tree of Emotions')
fig.show()