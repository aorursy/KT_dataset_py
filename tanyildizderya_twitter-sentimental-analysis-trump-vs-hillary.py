!pip install vaderSentiment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import matplotlib.pyplot as plt#visualization
%matplotlib inline
import seaborn as sns
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
#import data
tweets = pd.read_csv(r"../input/clinton-trump-tweets/tweets.csv")    
tweets.head()
tweets = tweets[[ 'handle', 'text', 'is_retweet', 'original_author', 
                 'time', 'lang', 'retweet_count', 'favorite_count']]
tweets.head()
tweets["lang"] = tweets[tweets["lang"] == "en"]
tweets = tweets[[ 'handle', 'text', 'is_retweet','time', 'lang', 'retweet_count', 'favorite_count']]
from datetime import datetime
date_format = "%Y-%m-%dT%H:%M:%S" 
tweets["time"]   = pd.to_datetime(tweets["time"],format = date_format)
tweets["hour"]   = pd.DatetimeIndex(tweets["time"]).hour
tweets["month"]  = pd.DatetimeIndex(tweets["time"]).month
tweets["day"]    = pd.DatetimeIndex(tweets["time"]).day
tweets["month_f"]  = tweets["month"].map({1:"JAN",2:"FEB",3:"MAR",
                                        4:"APR",5:"MAY",6:"JUN",
                                        7:"JUL",8:"AUG",9:"SEP"})
#trump tweets without retweets
tweets_trump = (tweets[(tweets["handle"] == "realDonaldTrump") &
                         (tweets["is_retweet"] == False)].reset_index()
                  .drop(columns = ["index"],axis = 1))
tweets_trump.head()
#hillary tweets without retweets
tweets_hillary = (tweets[(tweets["handle"] == "HillaryClinton") &
                            (tweets["is_retweet"] == False)].reset_index()
                              .drop(columns = ["index"],axis = 1))
tweets_hillary.head()
#Thanks @pavanraj159 !

plt.style.use('ggplot')

plt.figure(figsize = (13,6))
plt.subplot(121)
tweets[tweets["handle"] ==
       "realDonaldTrump"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                                wedgeprops = {"linewidth" : 1,
                                                                              "edgecolor" : "k"},
                                                                shadow = True,fontsize = 13,
                                                                explode = [.1,0.09],
                                                                startangle = 20,
                                                                colors = ["#FF3300","w"]
                                                               )
plt.ylabel("")
plt.title("Percentage of retweets - Trump")

plt.subplot(122)
tweets[tweets["handle"] ==
       "HillaryClinton"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                                wedgeprops = {"linewidth" : 1,
                                                                              "edgecolor" : "k"},
                                                                shadow = True,fontsize = 13,
                                                                explode = [.09,0],
                                                                startangle = 60,
                                                                colors = ["#6666FF","w"]
                                                               )
plt.ylabel("")
plt.title("Percentage of retweets - Hillary")
plt.show()
analyzer = SentimentIntensityAnalyzer()

def calculate_sentiment_scores(sentence):
    sntmnt = analyzer.polarity_scores(sentence)['compound']
    return(sntmnt)
eng_snt_score =  []

for comment in tweets_trump.text.to_list():
    snts_score = calculate_sentiment_scores(comment)
    eng_snt_score.append(snts_score)
tweets_trump['sentiment_score'] = np.array(eng_snt_score)
tweets_trump.head()
i = 0

vader_sentiment = [ ]

while(i<len(tweets_trump)):
    if ((tweets_trump.iloc[i]['sentiment_score'] >= 0.05)):
        vader_sentiment.append('positive')
        i = i+1
    elif ((tweets_trump.iloc[i]['sentiment_score'] > -0.05) & (tweets_trump.iloc[i]['sentiment_score'] < 0.05)):
        vader_sentiment.append('neutral')
        i = i+1
    elif ((tweets_trump.iloc[i]['sentiment_score'] <= -0.05)):
        vader_sentiment.append('negative')
        i = i+1
tweets_trump['vader_sentiment_labels'] = vader_sentiment
tweets_trump.vader_sentiment_labels.value_counts()
tweets_trump.head()
tweets_trump['vader_sentiment_labels'].value_counts().plot(kind='bar',figsize=(12,8));
eng_snt_score =  []

for comment in tweets_hillary.text.to_list():
    snts_score = calculate_sentiment_scores(comment)
    eng_snt_score.append(snts_score)
tweets_hillary['sentiment_score'] = np.array(eng_snt_score)
tweets_hillary.head()
i = 0

vader_sentiment = [ ]

while(i<len(tweets_hillary)):
    if ((tweets_hillary.iloc[i]['sentiment_score'] >= 0.05)):
        vader_sentiment.append('positive')
        i = i+1
    elif ((tweets_hillary.iloc[i]['sentiment_score'] > -0.05) & (tweets_hillary.iloc[i]['sentiment_score'] < 0.05)):
        vader_sentiment.append('neutral')
        i = i+1
    elif ((tweets_hillary.iloc[i]['sentiment_score'] <= -0.05)):
        vader_sentiment.append('negative')
        i = i+1
tweets_hillary['vader_sentiment_labels'] = vader_sentiment
tweets_hillary.vader_sentiment_labels.value_counts()
tweets_hillary.head()
tweets_hillary['vader_sentiment_labels'].value_counts().plot(kind='bar',figsize=(12,8));
from wordcloud import WordCloud

hsh_wrds_t = tweets_trump["text"].str.extractall(r'(\#\w+)')[0]
hsh_wrds_h = tweets_hillary["text"].str.extractall(r'(\#\w+)')[0]

def build_word_cloud(words,back_color,palette,title) :
    word_cloud = WordCloud(scale = 7,max_words = 1000,
                           max_font_size = 100,background_color = back_color,
                           random_state = 0,colormap = palette
                          ).generate(" ".join(words))
    plt.figure(figsize = (13,8))
    plt.imshow(word_cloud,interpolation = "bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

build_word_cloud(hsh_wrds_t,"black","rainbow","Hashtags - Trump")
build_word_cloud(hsh_wrds_h,"black","rainbow","Hashtags - Hillary")
