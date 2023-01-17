import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Getting data

tweets = pd.read_csv('../input/demonetization-tweets.csv', encoding = "ISO-8859-1")

print(tweets.head())
print(tweets['text'].str.upper().head())
print(tweets['text'][0])
#tweets['text'].str.split(': ',expand=True)[0]

#tweets['text'].str.contains('@', na=False).astype(int)

#tweets_bis=tweets['text'].str.replace('RT @', '@', case=True)

#print(tweets_bis)

#tweets_bis.str.startswith('@', na=False).astype(int)



#del RT @blablabla:



tweets['text_new'] = ''



import re



for i in range(len(tweets['text'])):

    m = re.search('(?<=:)(.*)', tweets['text'][i])    

    try:

        tweets['text_new'][i]=m.group(0)

    except AttributeError:

        tweets['text_new'][i]=tweets['text'][i]

        

print(tweets['text_new'])        
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



def wordcloud_by_province(tweets):

    stopwords = set(STOPWORDS)

    stopwords.add("https")

    stopwords.add("00A0")

    stopwords.add("00BD")

    stopwords.add("00B8")

    stopwords.add("ed")

    stopwords.add("demonetization")

    stopwords.add("Demonetization co")

    #Narendra Modi is the Prime minister of India

    stopwords.add("lakh")

    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets['text_new'].str.upper()]))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("Demonetization")



wordcloud_by_province(tweets)  
def wordcloud_by_province(tweets):

    a = pd.DataFrame(tweets['text'].str.contains("terrorists").astype(int))

    b = list(a[a['text']==1].index.values)

    stopwords = set(STOPWORDS)

    stopwords.add("https")

    stopwords.add("terrorists")

    stopwords.add("00A0")

    stopwords.add("00BD")

    stopwords.add("00B8")

    stopwords.add("ed")

    stopwords.add("demonetization")

    stopwords.add("Demonetization co")

    stopwords.add("lakh")

    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets.ix[b,:]['text_new'].str.upper()]))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("Tweets with word 'terrorists'")



wordcloud_by_province(tweets)  
def wordcloud_by_province(tweets):

    a = pd.DataFrame(tweets['text'].str.contains("narendramodi").astype(int))

    b = list(a[a['text']==1].index.values)

    stopwords = set(STOPWORDS)

    stopwords.add("narendramodi")

    stopwords.add("https")

    stopwords.add("00A0")

    stopwords.add("00BD")

    stopwords.add("00B8")

    stopwords.add("ed")

    stopwords.add("demonetization")

    stopwords.add("Demonetization co")

    stopwords.add("lakh")

    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets.ix[b,:]['text_new'].str.upper()]))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("Tweets with word 'narendramodi'")



wordcloud_by_province(tweets)  