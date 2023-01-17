import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from mpl_toolkits.basemap import Basemap

from wordcloud import WordCloud
import squarify
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re 
import gensim 
from gensim import corpora

%matplotlib inline
plt.style.use('fivethirtyeight')
plt.style.use('bmh')
#path = 'file/'
path = '../input/'
business = pd.read_csv(path + 'yelp_business.csv')
business_hour = pd.read_csv(path + 'yelp_business_hours.csv')
checkin = pd.read_csv(path + 'yelp_checkin.csv')
tips = pd.read_csv(path+'yelp_tip.csv',nrows = 10000)
review = pd.read_csv(path+'yelp_review.csv',nrows = 10000)
def basic_details(df):
    print('Row:{}, columns:{}'.format(df.shape[0],df.shape[1]))
    k = pd.DataFrame()
    k['number of Unique value'] = df.nunique()
    k['Number of missing value'] = df.isnull().sum()
    k['Data type'] = df.dtypes
    return k
business.head()
basic_details(business)
plt.figure(figsize=(12,4))
ax = sns.countplot(business['stars'])
plt.title('Distribution of rating');
f,ax = plt.subplots(1,2, figsize=(14,8))
ax1,ax2, = ax.flatten()
cnt = business['name'].value_counts()[:20].to_frame()

sns.barplot(cnt['name'], cnt.index, palette = 'RdBu', ax =ax1)
ax1.set_xlabel('')
ax1.set_title('Top name of store in Yelp')

cnt = business['neighborhood'].value_counts()[:20].to_frame()

sns.barplot(cnt['neighborhood'], cnt.index, palette = 'rainbow', ax =ax2)
ax2.set_xlabel('')
ax2.set_title('Top Neighborhood location')
plt.subplots_adjust(wspace=0.3)
gc.collect()
fig = plt.figure(figsize=(14, 8), edgecolor='w')
m = Basemap(projection='cyl',llcrnrlon= -180, urcrnrlon = 180, llcrnrlat = -90, urcrnrlat= 90,resolution='c',
           lat_ts = True)
m.drawcoastlines()
m.fillcontinents(color='#04BAE3',lake_color='#FFFFFF')
m.drawcountries()
m.drawmapboundary(fill_color='#FFFFFF')

mloc = m(business['latitude'].tolist(),business['longitude'].tolist())
m.scatter(mloc[1],mloc[0],color ='red',lw=3,alpha=0.3,zorder=5)
print('Number of city listed',business['city'].nunique())
f,ax = plt.subplots(1,2, figsize=(14,8))
ax1,ax2, = ax.flatten()
cnt = business['city'].value_counts()[:20].to_frame()

sns.barplot(cnt['city'], cnt.index, palette = 'gist_rainbow', ax =ax1)
ax1.set_xlabel('')
ax1.set_title('Top city business listed in Yelp')

cnt = business['state'].value_counts()[:20].to_frame()

sns.barplot(cnt['state'], cnt.index, palette = 'coolwarm', ax =ax2)
ax2.set_xlabel('')
ax2.set_title('Top state business listed in Yelp');
print('Median review count',business['review_count'].median())
plt.figure(figsize = (14,10))
sns.barplot(business[business['review_count'] >3000]['review_count'],business[business['review_count'] >3000]['name'],
           palette = 'summer')
plt.xlabel('')
plt.title('Top review count');
plt.figure(figsize=(14,5))
sns.countplot(business['is_open'])
cloud = WordCloud(width=1440, height= 1080,max_words= 1000).generate(' '.join(business['categories'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off');
print('Maximum number of category',business['categories'].str.split(';').str.len().max())
print('Median category of business',business['categories'].str.split(';').str.len().median())
corpus = ' '.join(business['categories'])

corpus = pd.DataFrame(corpus.split(';'),columns=['categories'])
cnt = corpus['categories'].value_counts().to_frame()[:20]
plt.figure(figsize=(14,8))
sns.barplot(cnt['categories'], cnt.index, palette = 'tab20')
plt.title('Top main categories listing');
checkin.head()
basic_details(checkin)
f,ax = plt.subplots(1,2, figsize = (14,6))
ax1,ax2, = ax.flatten()
cnt = checkin['weekday'].value_counts().to_frame()
sns.barplot(cnt['weekday'], cnt.index, palette = 'ocean', ax=ax1)
ax1.set_title('Distribution of weekday')
ax1.set_xlabel('')

cnt = checkin['hour'].value_counts().to_frame()
sns.barplot(cnt['hour'], cnt.index, palette = 'hot', ax=ax2)
ax2.set_title('Distribution of Hour')
ax2.set_xlabel('');
k = checkin.groupby(['weekday','hour',])['checkins'].sum().to_frame().reset_index()

plt.figure(figsize=(14,6))
sns.pointplot(y = k['checkins'],x = k['hour'],hue = k['weekday'],alpha=0.3)
plt.ylabel('Checkins')
plt.title('Distribution of checkins on different weekday (log)')
plt.xlabel('Weekday')
plt.xticks(rotation=45);
tips.head()
basic_details(tips)
# Word cloud
cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(tips['text'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off');
# Meta feature of text
tips['num_words'] = tips['text'].str.len()
tips['num_uniq_words'] = tips['text'].apply(lambda x: len(set(str(x).split())))
tips['num_chars'] = tips['text'].apply(lambda x: len(str(x)))
tips['num_stopwords'] = tips['text'].apply(lambda x: len([w for w in str(x).lower().split() 
                                                      if w in set(stopwords.words('english'))]))
# Distribution of text feature
f, ax = plt.subplots(2,2, figsize = (14,10))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.distplot(tips['num_words'],bins=100,color='r', ax=ax1)
ax1.set_title('Distribution of Number of words')

sns.distplot(tips['num_uniq_words'],bins=100,color='b', ax=ax2)
ax2.set_title('Distribution of Unique words')

sns.distplot(tips['num_chars'],bins=100,color='y', ax=ax3)
ax3.set_title('Distribution of Char words')

sns.distplot(tips['num_stopwords'],bins=100,color='r', ax=ax4)
ax4.set_title('Distribution of Stop words')

tips['date'] = pd.to_datetime(tips['date'])
tips['year'] = tips['date'].dt.year
tips['month'] = tips['date'].dt.month

f,ax = plt.subplots(1,2,figsize = (14,6))
ax1,ax2 = ax.flatten()
cnt  = tips.groupby('year').sum()['likes'].to_frame()
sns.barplot(cnt.index,cnt['likes'],palette='hot', ax = ax1)
ax1.set_title('Distribution of stars by year')
ax1.set_ylabel('')

cnt  = tips.groupby('month').sum()['likes'].to_frame()
sns.barplot(cnt.index,cnt['likes'],palette='ocean', ax = ax2)
ax2.set_title('Distribution of stars by month')
ax2.set_ylabel('')
# clean text
lemma = WordNetLemmatizer()

def clean_text(doc):
    corpus = []
    for c in range(0, doc.shape[0]):
        stop_free = ' '.join([i for i in doc['text'][c].lower().split() if i not in set(stopwords.words('english'))])
        puct_free = ''.join(i for i in stop_free if i not in set(string.punctuation))
        normalized = [lemma.lemmatize(word) for word in puct_free.split()]
        corpus.append(normalized)
    return corpus
doc_tips = clean_text(tips)
# LDA model
dictionary = corpora.Dictionary(doc_tips)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_tips]
ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics= 3, id2word= dictionary, passes=20)

#print(ldamodel.print_topics(num_topics=3,num_words=5))

for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words= 5):
    print('Topic {}: words'.format(topic[0]))
    topic_word = [w for (w,val) in topic[1]]
    print(topic_word)
# Top topics in document
tp = ldamodel.top_topics(doc_term_matrix,topn=20,dictionary=dictionary)
# tuple unpacking
label = [] 
value = []

f,ax = plt.subplots(1,2,figsize = (14,6))
ax1.set_title(tp[0])
ax1,ax2 = ax.flatten()
for i,k in tp[0][0]:
    label.append(i)
    value.append(k)
sns.barplot(label,value,palette='BrBG', ax=ax1)

label = [] 
value = []
for i,k in tp[1][0]:
    label.append(i)
    value.append(k)
sns.barplot(label,value,palette='RdBu_r', ax= ax2);
business_hour.head()
basic_details(business_hour)
# Business hour timing
col = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday','saturday', 'sunday']
op = business_hour.groupby(col).count().reset_index().sort_values(by='business_id',ascending=False)

op[:20]
review.head()
# Word cloud
cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(review['text'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off');
review['date'] = pd.to_datetime(review['date'])
review['year'] = review['date'].dt.year
review['month'] = review['date'].dt.month

f,ax = plt.subplots(1,2, figsize = (14,6))
ax1,ax2 = ax.flatten()
cnt = review.groupby('year').count()['stars'].to_frame()
sns.barplot(cnt.index, cnt['stars'],palette = 'gist_rainbow', ax=ax1)

for ticks in ax1.get_xticklabels():
    ticks.set_rotation(45)

cnt = review.groupby('month').count()['stars'].to_frame()
sns.barplot(cnt.index, cnt['stars'],palette = 'coolwarm', ax = ax2)
# clean text
lemma = WordNetLemmatizer()

def clean_text(doc):
    corpus = []
    for c in range(0, doc.shape[0]):
        stop_free = ' '.join([i for i in doc['text'][c].lower().split() if i not in set(stopwords.words('english'))])
        puct_free = ''.join(i for i in stop_free if i not in set(string.punctuation))
        normalized = [lemma.lemmatize(word) for word in puct_free.split()]
        corpus.append(normalized)
    return corpus
doc_review = clean_text(review)
# LDA model
dictionary = corpora.Dictionary(doc_review)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_review]
ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics= 3, id2word= dictionary, passes=20)

#print(ldamodel.print_topics(num_topics=3,num_words=5))

for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words= 5):
    print('Topic {}: words'.format(topic[0]))
    topic_word = [w for (w,val) in topic[1]]
    print(topic_word)
# Top topics in document
tp = ldamodel.top_topics(doc_term_matrix,topn=20,dictionary=dictionary)
# tuple unpacking
label = [] 
value = []

f,ax = plt.subplots(1,2,figsize = (14,6))
ax1.set_title(tp[0])
ax1,ax2 = ax.flatten()
for i,k in tp[0][0]:
    label.append(i)
    value.append(k)
sns.barplot(label,value,palette='BrBG', ax=ax1)

label = [] 
value = []
for i,k in tp[1][0]:
    label.append(i)
    value.append(k)
sns.barplot(label,value,palette='RdBu_r', ax= ax2);
## continued