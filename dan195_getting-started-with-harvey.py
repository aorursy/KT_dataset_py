import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv('../input/Tropical_Storm_Harvey.csv',encoding="ISO-8859-1")

data_hurr = pd.read_csv('../input/Hurricane_Harvey.csv',encoding="ISO-8859-1")



data.shape,list(data)
data.tail()
del data['ID']

del data_hurr['ID']



data.describe()
data.isnull().sum()
data_hurr.isnull().sum()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



list_stops = ('hurricane','harvey','hurricane harvey','tropical',"tropical storm",'hurricaneharvey','https','twitter',"tt",

             "goo","bit","ly","bit")



for word in list_stops:

    STOPWORDS.add(word)



tweets =data['Tweet'].astype("str")

tweets =''.join(tweets)



wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(tweets)

plt.imshow(wordcloud)

plt.title('Frequent words for recent Hurricane Harvey tweets')

plt.axis('off')

plt.show()
tweets = data_hurr["Tweet"].astype("str").dropna()

tweets =''.join(tweets)



wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(tweets)

plt.imshow(wordcloud)

plt.title('Frequent words for recent Hurricane Harvey tweets')

plt.axis('off')

plt.show()