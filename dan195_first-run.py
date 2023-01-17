



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import nltk



df = pd.read_csv('../input/comcast_consumeraffairs_complaints.csv')





df_fcc = pd.read_csv("../input/comcast_fcc_complaints_2015.csv")



df.head()
print(df.shape,df_fcc.shape)
sns.countplot(x='rating',data=df,palette='Blues_d')
df.rating.value_counts()
df_no_rating = df.loc[df.rating == 0]

df = df.loc[df.rating != 0]

print(df_no_rating.shape,df.shape)
sns.countplot(x='rating',data=df,palette='Blues_d')
Positive = int(df['rating'].loc[df['rating'] >= 3].count())

Negative = int(df['rating'].loc[df['rating'] < 3].count())

print("There were {} positive reviews and at least {} negative reviews.".format(str(Positive),str(Negative)))
df['state'] = df['author'].str[-2:].apply(lambda x: x.upper())

df.state.value_counts()
(df.loc[df.rating > 2].state.value_counts() / df.state.value_counts()).sort_values(ascending=False)
from wordcloud import WordCloud, STOPWORDS



list_stops = ('comcast','time','customer','even','now','company',

            'day','someone','thing','also','got','way','call','called','one','said','tell','service')



for word in list_stops:

    STOPWORDS.add(word)
low_ratings =df['text'].dropna().loc[df['rating']<3].tolist()

low_ratings =''.join(low_ratings).lower()



high_ratings =df['text'].dropna().loc[df['rating']>=3].tolist()

high_ratings =''.join(high_ratings).lower()





no_ratings = df_no_rating['text'].dropna().tolist()

no_ratings =''.join(no_ratings).lower()
wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(low_ratings)

plt.imshow(wordcloud)

plt.title('Frequent words amongst low ratings')

plt.axis('off')

plt.show()
wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(high_ratings)

plt.imshow(wordcloud)

plt.title('Frequent words amongst high ratings')

plt.axis('off')

plt.show()
high_ratings =df['text'].dropna().loc[df['rating']>=4].tolist()

high_ratings =''.join(high_ratings).lower()



wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(high_ratings)

plt.imshow(wordcloud)

plt.title('Frequent words amongst high ratings')

plt.axis('off')

plt.show()
counts = df_fcc.State.value_counts()

ax = counts.iloc[:10].plot(kind="barh")

ax.invert_yaxis()
counts = df_fcc.City.value_counts()

ax = counts.iloc[:20].plot(kind="barh")

ax.invert_yaxis()
df_fcc['Customer Complaint'].value_counts()
from wordcloud import STOPWORDS

common_complaints = df_fcc['Customer Complaint'].dropna().tolist()

common_complaints =''.join(common_complaints).lower()



list_stops = ('comcast','now','company','day','someone','thing','also','got','way','call','called','one','said','tell')



for word in list_stops:

    STOPWORDS.add(word)

    

  

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(common_complaints)

plt.imshow(wordcloud)

plt.title('Frequent words for customer complaints')

plt.axis('off')

plt.show()