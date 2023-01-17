import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
pd.set_option('max_columns', None)

%matplotlib inline
df = pd.read_csv('../input/booking_com-travel_sample.csv')
df.head()
df.describe()
df.info()
len(df['city'].unique())
df['city'].value_counts().head()
df['hotel_star_rating'].value_counts()
def star_cleaning(value):
    try:
        value = list(value)[0]
        if int(value) == 3:
            return (3.0)

        elif int(value) == 2:
            return (2.0)

        elif int(value) == 1:
            return (1.0)

        elif int(value) == 4:
            return (4.0)
        else:
            return (5.0)
        
    except:
        pass
df['hotel_star_rating'] = df['hotel_star_rating'].apply(star_cleaning)
df['room_count'].describe()
room_count = df['room_count']
room_count.dropna(inplace = True)
plt.figure(figsize=(15,10))
plt.title('Distribution Plot', fontsize=18)
plt.xlabel('xlabel',fontsize=18)
sns.distplot(df['room_count'], color='g')
plt.figure(figsize=(15, 5))
plt.title('Box Plot', fontsize=18)
plt.xlabel('xlabel',fontsize=18)
sns.boxplot(df['room_count'])
room_count = df['hotel_star_rating']
room_count.dropna(inplace = True)
plt.figure(figsize=(15, 5))
plt.title('Count Plot', fontsize=18)
plt.xlabel('xlabel',fontsize=18)
sns.countplot(df['hotel_star_rating'])
plt.figure(figsize = (15,10))
plt.title('Count Plot (Hotels by States)', fontsize=18)
plt.xticks(rotation = (90))
plt.xlabel('xlabel',fontsize=18)
sns.countplot(df['state'], palette="Set2")
df['hotel_facilities'][0].split('•')
df['hotel_facilities'][0].split('•')[1].split('|')
' '.join(list(itertools.chain.from_iterable([re.sub(r'[^\w\s]',' ',x.replace(':', ' ').lower().strip()).split('|') for x in df['hotel_facilities'][10].split('•')]))).replace('\n','')
def one_liner(val):
    try:
        return(' '.join(list(itertools.chain.from_iterable([re.sub(r'[^\w\s]',' ',x.replace(':', ' ').lower().strip()).split('|') for x in val.split('•')]))).replace('\n',''))
    except:
        pass
    
df['hotel_facilities_wc'] = df['hotel_facilities'].apply(one_liner)
wl = []
for hotel_fac in list(filter(None, df['hotel_facilities_wc'])):
    wl.append(hotel_fac)
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='White', max_words=50 ,collocations=False).generate(' '.join(wl))
plt.figure(figsize = (10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
