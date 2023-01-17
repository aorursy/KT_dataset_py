# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt
t_data = pd.read_csv("../input/Tweets.csv")

#first five rows of data

t_data.head()
def reason_each_flight(airline):

    data = t_data[t_data['airline'] == airline]

    data = data['negativereason']

    data_count = data.value_counts()

    List = data.value_counts().index.tolist()

    Index = range(1,(len(data.unique())))

    plt.bar(Index, data_count)

    plt.xlabel('Negative Reason')

    plt.ylabel('Total Number for Negative Reason')

    plt.title('Total number for every negative reason for ' + airline)

    plt.xticks(Index, List,rotation = 90)

    

Air = t_data['airline'].value_counts().index.tolist()

plt.figure(1,figsize=(15, 25))

plt.subplot(321)

reason_each_flight(Air[0])

plt.subplot(322)

reason_each_flight(Air[1])

plt.subplot(323)

reason_each_flight(Air[2])

plt.subplot(324)

reason_each_flight(Air[3])

plt.subplot(325)

reason_each_flight(Air[4])

plt.subplot(326)

reason_each_flight(Air[5])

plt.tight_layout()    
#let's make Word Cloud for negative, neutral, positive mood

#import wordcloud and stopwords

from wordcloud import WordCloud,STOPWORDS

df=t_data[t_data['airline_sentiment']=='negative'] 

# join tweets to a single string

words = ' '.join(df['text'])

# remove URLs, RTs, and twitter handles

no_urls_no_tags = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2000,

                      height=1500

                     ).generate(no_urls_no_tags)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df=t_data[t_data['airline_sentiment']=='neutral'] 

# join tweets to a single string

words = ' '.join(df['text'])

# remove URLs, RTs, and twitter handles

no_urls_no_tags = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2000,

                      height=1500

                     ).generate(no_urls_no_tags)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df=t_data[t_data['airline_sentiment']=='positive'] 

# join tweets to a single string

words = ' '.join(df['text'])

# remove URLs, RTs, and twitter handles

no_urls_no_tags = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2000,

                      height=1500

                     ).generate(no_urls_no_tags)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()