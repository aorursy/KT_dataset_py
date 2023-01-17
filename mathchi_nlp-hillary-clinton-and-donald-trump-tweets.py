# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib.pyplot as plt#visualization

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

tweets = pd.read_csv("../input/clinton-trump-tweets/tweets.csv")

tweets = tweets[[ 'handle', 'text', 'is_retweet', 'original_author', 

                 'time', 'lang', 'retweet_count', 'favorite_count']]
tweets.head()
tweets['lang'].value_counts()
def language(df) :

    if df["lang"] == "en" :

        return "English"

    elif df["lang"] == "es" :

        return "Spanish"

    else :

        return "Other"



tweets["lang"] = tweets.apply(lambda tweets:language(tweets),axis = 1)





# datetime convert

from datetime import datetime

date_format = "%Y-%m-%dT%H:%M:%S" 

tweets["time"]   = pd.to_datetime(tweets["time"],format = date_format)

tweets["hour"]   = pd.DatetimeIndex(tweets["time"]).hour

tweets["month"]  = pd.DatetimeIndex(tweets["time"]).month

tweets["day"]    = pd.DatetimeIndex(tweets["time"]).day

tweets["month_f"]  = tweets["month"].map({1:"JAN",2:"FEB",3:"MAR",

                                        4:"APR",5:"MAY",6:"JUN",

                                        7:"JUL",8:"AUG",9:"SEP"})
tweets['lang'].value_counts()
data=pd.concat([tweets.handle, tweets.text], axis=1)       # data icinde handle ve text kisimini ayirip concate edip DF yaptik. cunku diger kisimlara ihtiyacimiz olmayacak
data.dropna(axis=0, inplace=True)                  # bos gozlem yerlerinin satirlarini sildik
data.handle.value_counts()  
#Total number of tweets by both of the twitter handles

sns.countplot(x='handle', data = tweets)
#Number of tweets by the months

monthly_tweets = tweets.groupby(['month', 'handle']).size().unstack()

monthly_tweets.plot(title='Monthly Tweet Counts', colormap='copper')
#trump tweets without retweets

tweets_trump   = (tweets[(tweets["handle"] == "realDonaldTrump") &

                         (tweets["is_retweet"] == False)].reset_index()

                  .drop(columns = ["index"],axis = 1))



#trump tweets with retweets

tweets_trump_retweets   = (tweets[(tweets["handle"] == "realDonaldTrump") &

                                  (tweets["is_retweet"] == True)].reset_index()

                                  .drop(columns = ["index"],axis = 1))



#hillary tweets without retweets

tweets_hillary  = (tweets[(tweets["handle"] == "HillaryClinton") &

                            (tweets["is_retweet"] == False)].reset_index()

                              .drop(columns = ["index"],axis = 1))



#hillary tweets with retweets

tweets_hillary_retweets  = (tweets[(tweets["handle"] == "HillaryClinton") &

                            (tweets["is_retweet"] == True)].reset_index()

                              .drop(columns = ["index"],axis = 1))
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

                                                                colors = ["#ff0026","#fbff00"]

                                                               )

plt.ylabel("")

plt.title("Percentage of Trump retweets")





plt.subplot(122)

tweets[tweets["handle"] ==

       "HillaryClinton"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",

                                                                wedgeprops = {"linewidth" : 1,

                                                                              "edgecolor" : "k"},

                                                                shadow = True,fontsize = 13,

                                                                explode = [.09,0],

                                                                startangle = 60,

                                                                colors = ["#0095ff","#fbff00"]

                                                               )

plt.ylabel("")

plt.title("Percentage of Hillary retweets")

plt.show()
# regular expression yapalim. Burada gulucuk ve benzeri ifadeleri silmek icin



import re



first_text = data.text[5]                   # butun datadan once 5.siradaki datanin temizlenmesini yapalim bakalim temizlenmis ise butun dataya uygulamaya calisacagiz

text = re.sub("[^a-zA-Z]"," ", first_text)

text = text.lower()                        # ilk ve son haline gore description icindeki elemanlari kucultmesini istedik ve yazdirirsak kuculdugunu gorebiliriz
text
import nltk         # nltk nin download kiti bulunmaktadir. Bisey ifade etmeyen kelimeleri(Stop words ler ornegin 'an' veya 'the') cikarmamiz gerekiyor. Eger cikarmazsak 'an' veya 'the' kelimeleri enfazla kullanilanlar olarak gorunmus olacak bunlar bizim icin bisey ifade etmiyor

nltk.download("stopwords")          # hazir olarak bulunan stopwordslari indirip uygulayabiliriz

nltk.download("punkt")



from nltk.corpus import stopwords
text = nltk.word_tokenize(text)       # tokenize ettik yani kelimeleri birbirinden ayirdik. ayirmak icin split kullandik cunku mesela "doesn't" vy "shouldn't" kelimesini "does" ve "not" olarak ayiramiyor bunun icin tokenize kullandik

text = [ word for word in text if not word in set(stopwords.words("english"))]
text
# kelimeleri sade haline indirgemek icin



import nltk as nlp

nltk.download('wordnet')



lemma = nlp.WordNetLemmatizer()

text = [lemma.lemmatize(word) for word in text]



text = " ".join(text)
# simdi BUTUN VERI dekilere bunu uygulamak icin for dongusu olusturalim:



text_list = []

for text in data.text:

    text = re.sub("[^a-zA-Z]"," ", text)

    text= text.lower()

    text= nltk.word_tokenize(text)

    text = [ word for word in text if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    text = [lemma.lemmatize(word) for word in text]

    text = " ".join(text)

    text_list.append(text)
text_list
# bag of words: kac kelime kullanmak istiyorsak kendimiz belirliyoruz. Duygu kelimelerini verip analizini yapacagiz



from sklearn.feature_extraction.text import CountVectorizer        # bag of words olusturmak icin 



max_features = 5000               # max ... kadar kelimeye baksin



# Simdi modelimizi olusturalim

count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(text_list).toarray()                     # modelimizi fitleyip array donusturduk

print(f"{max_features} most used words:\n\n{count_vectorizer.get_feature_names()}")
plt.figure(figsize = (12,8))

sns.countplot(x = "month_f",hue = "handle",palette = ["#ff0026","#0095ff"],

              data = tweets.sort_values(by = "month",ascending = True),

             linewidth = 1,edgecolor = "k"*tweets_trump["month"].nunique())

plt.grid(True)

plt.title("Tweets month by month in 2016 election")

plt.show()
from textblob import TextBlob



bloblist_desc = list()                                  # butun tweet ler listeleniyor



df_tweet_descr_str=tweets['text'].astype(str)           # text ler ayiklaniyor



for row in df_tweet_descr_str:

    blob = TextBlob(row)

    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))

    df_tweet_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])

 

def f(df_tweet_polarity_desc):

    if df_tweet_polarity_desc['sentiment'] > 0:

        val = "Positive"

    elif df_tweet_polarity_desc['sentiment'] == 0:

        val = "Neutral"

    else:

        val = "Negative"

    return val



df_tweet_polarity_desc['Sentiment_Type'] = df_tweet_polarity_desc.apply(f, axis=1)



plt.figure(figsize=(10,10))

sns.set_style("whitegrid")

ax = sns.countplot(x="Sentiment_Type", data=df_tweet_polarity_desc)