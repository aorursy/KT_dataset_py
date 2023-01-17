import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import re

import seaborn as sns

from IPython.display import display

pd.options.mode.chained_assignment = None

import matplotlib

matplotlib.style.use('ggplot')



#Getting data

tweets = pd.read_csv('../input/demonetization-tweets.csv', encoding = "ISO-8859-1")

display(tweets.head(3))
import re

#Preprocessing del RT @blablabla:

tweets['text_new'] = ''

tweets['tweetos'] = '' 



#add tweetos first part

for i in range(len(tweets['text'])):

    try:

        tweets['tweetos'][i] = tweets['text'].str.split(':')[i][0]

    except AttributeError:    

        tweets['tweetos'][i] = 'other'



#Preprocessing tweetos. select tweetos contains 'RT @'

for i in range(len(tweets['text'])):

    if tweets['tweetos'].str.contains('RT @')[i]  == False:

        tweets['tweetos'][i] = 'other'



#'text_new' is the feature 'text' without the tweetos    

for i in range(len(tweets['text'])):

    m = re.search('(?<=:)(.*)', tweets['text'][i])

    if tweets['text'].str.contains('RT @')[i]  == True:

        try:

            tweets['text_new'][i]=m.group(0)

        except AttributeError:

            tweets['text_new'][i]=tweets['text'][i] 

    else:       

        tweets['text_new'][i] =  tweets['text'][i]       
#tweets['text_new_bis'] = tweets['text_new'].str.contains(r'^https?:\/\/.*[\r\n]*')

#for i in range(len(tweets['text'])):

#    m =  re.split('https', tweets['text_new'][i])

#    #tweets['text_new_bis'][i]

#    try:

#        print(m[1])

#    except IndexError:  

#        print('')

#print(tweets['text_new_bis'][0])

#print(tweets['text_new_bis'])

#print(tweets['text_new'][7999])
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
print(tweets['retweetCount'].describe())
tweets['nb_words'] = 0

for i in range(len(tweets['text'])):

    tweets['nb_words'][i] = len(tweets['text'][i].split(' '))
tweets['hour'] = pd.DatetimeIndex(tweets['created']).hour

tweets['date'] = pd.DatetimeIndex(tweets['created']).date

tweets['minute'] = pd.DatetimeIndex(tweets['created']).minute
tweets_hour = tweets.groupby(['hour'])['retweetCount'].sum()

tweets_minute = tweets.groupby(['minute'])['retweetCount'].sum()

tweets['text_len'] = tweets['text'].str.len()

tweets_avgtxt_hour = tweets.groupby(['hour'])['text_len'].mean()

tweets_avgwrd_hour = tweets.groupby(['hour'])['nb_words'].mean()
import seaborn as sns

tweets_hour.transpose().plot(kind='line',figsize=(6.5, 4))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('The number of retweet by hour', bbox={'facecolor':'0.8', 'pad':0})
tweets_minute.transpose().plot(kind='line',figsize=(6.5, 4))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('The number of retweet by minute', bbox={'facecolor':'0.8', 'pad':0})
tweets_avgtxt_hour.transpose().plot(kind='line',figsize=(6.5, 4))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('The Average of lenght by hour', bbox={'facecolor':'0.8', 'pad':0})
tweets_avgwrd_hour.transpose().plot(kind='line',figsize=(6.5, 4))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('The Average number of words by hour', bbox={'facecolor':'0.8', 'pad':0})
#print(get_corpus(tweets['text']))
def get_stop_words(s, n):

	'''

	:s : pd.Series; each element as a list of words from tokenization

	:n : int; n most frequent words are judged as stop words 



	:return : list; a list of stop words

	'''

	from collections import Counter

	l = get_corpus(s)

	l = [x for x in Counter(l).most_common(n)]

	return l



def get_corpus(s):

	'''

	:s : pd.Series; each element as a list of words from tokenization



	:return : list; corpus from s

	'''

	l = []

	s.map(lambda x: l.extend(x))

	return l



#freqwords = get_stop_words(tweets['text'],n=60)



#freq = [s[1] for s in freqwords]



#plt.title('frequency of top 60 most frequent words', bbox={'facecolor':'0.8', 'pad':0})

#plt.plot(freq)

#plt.xlim([-1,60])

#plt.ylim([0,1.1*max(freq)])

#plt.ylabel('frequency')

#plt.show()
tweets['statusSource_new'] = ''



for i in range(len(tweets['statusSource'])):

    m = re.search('(?<=>)(.*)', tweets['statusSource'][i])

    try:

        tweets['statusSource_new'][i]=m.group(0)

    except AttributeError:

        tweets['statusSource_new'][i]=tweets['statusSource'][i]

        

#print(tweets['statusSource_new'].head())   



tweets['statusSource_new'] = tweets['statusSource_new'].str.replace('</a>', ' ', case=False)
tweets['statusSource_new'] = tweets['statusSource_new'].str.replace('</a>', ' ', case=False)

#print(tweets[['statusSource_new','retweetCount']])



tweets_by_type= tweets.groupby(['statusSource_new'])['retweetCount'].sum()

#print(tweets_by_type)
tweets_by_type.transpose().plot(kind='bar',figsize=(10, 5))

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Number of retweetcount by Source', bbox={'facecolor':'0.8', 'pad':0})
tweets['statusSource_new2'] = ''



for i in range(len(tweets['statusSource_new'])):

    if tweets['statusSource_new'][i] not in ['Twitter for Android ','Twitter Web Client ','Twitter for iPhone ']:

        tweets['statusSource_new2'][i] = 'Others'

    else:

        tweets['statusSource_new2'][i] = tweets['statusSource_new'][i] 

#print(tweets['statusSource_new2'])       



tweets_by_type2 = tweets.groupby(['statusSource_new2'])['retweetCount'].sum()
tweets_by_type2.rename("",inplace=True)

explode = (0, 0, 0, 1.0)

tweets_by_type2.transpose().plot(kind='pie',figsize=(6.5, 4),autopct='%1.1f%%',shadow=True,explode=explode)

plt.legend(bbox_to_anchor=(1, 1), loc=6, borderaxespad=0.)

plt.title('Number of retweetcount by Source bis', bbox={'facecolor':'0.8', 'pad':5})
from sklearn.feature_extraction.text import TfidfVectorizer

####

from nltk.stem import WordNetLemmatizer

#tweets['text_sep'] = [''.join(z).strip() for z in tweets['text_new']]

tweets['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweets['text_new']]       

####

vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)

X = vectorizer.fit_transform(tweets['text_lem'].str.upper())

print(X.shape)

#print(tweets['text_sep'])

#print(tweets['text_new'])
from sklearn.cluster import KMeans

km = KMeans(n_clusters=5,init='k-means++',max_iter=200,n_init=1)
km.fit(X)

terms = vectorizer.get_feature_names()

order_centroids = km.cluster_centers_.argsort()[:,::-1]

for i in range(5):

    print("cluster %d:" %i, end='')

    for ind in order_centroids[i,:10]:

        print(' %s' % terms[ind], end='')

    print()    
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pos = pca.fit_transform(dist)

xs, ys = pos[:,0], pos[:,1]
#set up colors per clusters using a dict

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}#, 5: '#8A2BE2', 6: '#E9967A'}

#8A2BE2

##E9967A

#set up cluster names using a dict

cluster_names = {0: 'cluster 1', 

                 1: 'cluster 2', 

                 2: 'cluster 3', 

                 3: 'cluster 4', 

                 4: 'cluster 5'}

                 #5: 'cluster 6',

                 #6: 'cluster 7'}

clusters = km.labels_.tolist()
#some ipython magic to show the matplotlib plots inline

%matplotlib inline 



#create data frame that has the result of the MDS plus the cluster numbers and titles

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title='')) 



#group by cluster

groups = df.groupby('label')





# set up plot

fig, ax = plt.subplots(figsize=(10, 4)) # set size

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling



#iterate through groups to layer the plot

#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label

for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, 

            label=cluster_names[name], color=cluster_colors[name],

            mec='none')

    ax.set_aspect('auto')

    ax.tick_params(\

        axis= 'x',          # changes apply to the x-axis

        which='both',      # both major and minor ticks are affected

        bottom='off',      # ticks along the bottom edge are off

        top='off',         # ticks along the top edge are off

        labelbottom='off')

    ax.tick_params(\

        axis= 'y',         # changes apply to the y-axis

        which='both',      # both major and minor ticks are affected

        left='off',      # ticks along the bottom edge are off

        top='off',         # ticks along the top edge are off

        labelleft='off')

    

ax.legend(numpoints=1)  #show legend with only 1 point

plt.title('Cluster plotting with ACP', bbox={'facecolor':'0.8', 'pad':0})

#add label in x,y position with the label as the film title

for i in range(len(df)):

    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=5)  



    

    

plt.show() #show the plot
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

tweets['favorited'] = le.fit_transform(tweets['favorited'])

tweets['replyToSN'] = tweets['replyToSN'].fillna(-999)

tweets['truncated'] = le.fit_transform(tweets['truncated'])

tweets['replyToSID'] = tweets['replyToSID'].fillna(-999)

tweets['id'] = le.fit_transform(tweets['id'])

tweets['replyToUID'] = tweets['replyToUID'].fillna(-999)

tweets['statusSource_new'] = le.fit_transform(tweets['statusSource_new'])

tweets['isRetweet'] = le.fit_transform(tweets['isRetweet'])

tweets['retweeted'] = le.fit_transform(tweets['retweeted'])

tweets['screenName'] = le.fit_transform(tweets['screenName'])

tweets['tweetos'] = le.fit_transform(tweets['tweetos'])



tweets_num = tweets[tweets.select_dtypes(exclude=['object']).columns.values]

tweets_num.drop('Unnamed: 0',inplace=True,axis=1)

tweets_num.drop('retweeted',inplace=True,axis=1)

tweets_num.drop('favorited',inplace=True,axis=1)

print(tweets.select_dtypes(exclude=['object']).columns.values)
#from string import letters

import seaborn as sns



sns.set(style="white")

# Compute the correlation matrix

corr = tweets_num.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 4))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(920, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.1,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.title('Correlation between numerical features', bbox={'facecolor':'0.8', 'pad':0})
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *



from nltk import tokenize



sid = SentimentIntensityAnalyzer()



tweets['sentiment_compound_polarity']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['compound'])

tweets['sentiment_neutral']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['neu'])

tweets['sentiment_negative']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['neg'])

tweets['sentiment_pos']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['pos'])

tweets['sentiment_type']=''

tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'

tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'

tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'

tweets.head()
import matplotlib

matplotlib.style.use('ggplot')



tweets_sentiment = tweets.groupby(['sentiment_type'])['sentiment_neutral'].count()

tweets_sentiment.rename("",inplace=True)

explode = (0, 0, 1.0)

plt.subplot(221)

tweets_sentiment.transpose().plot(kind='barh',figsize=(10, 6))

plt.title('Sentiment Analysis 1', bbox={'facecolor':'0.8', 'pad':0})

plt.subplot(222)

tweets_sentiment.plot(kind='pie',figsize=(10, 6),autopct='%1.1f%%',shadow=True,explode=explode)

plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)

plt.title('Sentiment Analysis 2', bbox={'facecolor':'0.8', 'pad':0})

plt.show()
tweets['count'] = 1

tweets_filtered = tweets[['hour', 'sentiment_type', 'count']]

pivot_tweets = tweets_filtered.pivot_table(tweets_filtered, index=["sentiment_type", "hour"], aggfunc=np.sum)

print(pivot_tweets.head())
sentiment_type = pivot_tweets.index.get_level_values(0).unique()

#f, ax = plt.subplots(2, 1, figsize=(8, 10))

plt.setp(ax, xticks=list(range(0,24)))



for sentiment_type in sentiment_type:

    split = pivot_tweets.xs(sentiment_type)

    split["count"].plot( legend=True, label='' + str(sentiment_type))

plt.title('Evolution of sentiments by hour', bbox={'facecolor':'0.8', 'pad':0})    
from  sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from xgboost import plot_importance



tweets_num_mod = tweets[tweets.select_dtypes(exclude=['object']).columns.values]

target = tweets_num_mod['retweetCount']

tweets_num_mod.drop('retweetCount',inplace=True,axis=1)

tweets_num_mod.drop('Unnamed: 0',inplace=True,axis=1)



#Just simple  and single model

model_xg = XGBRegressor()

model_rf = RandomForestRegressor()

model_et = ExtraTreesRegressor()

model_gb = GradientBoostingRegressor()

model_dt = DecisionTreeRegressor()
scores_xg = cross_val_score(model_xg, tweets_num_mod, target, cv=5,scoring='r2')

scores_rf = cross_val_score(model_rf, tweets_num_mod, target, cv=5,scoring='r2')

scores_dt = cross_val_score(model_dt, tweets_num_mod, target, cv=5,scoring='r2')

scores_et = cross_val_score(model_et, tweets_num_mod, target, cv=5,scoring='r2')

scores_gb = cross_val_score(model_gb, tweets_num_mod, target, cv=5,scoring='r2')
print("Mean of scores for XG:", sum(scores_xg) / float(len(scores_xg)))

print("Mean of scores for RF:", sum(scores_rf) / float(len(scores_rf)))

print("Mean of scores for DT:", sum(scores_dt) / float(len(scores_dt)))

print("Mean of scores for ET:", sum(scores_et) / float(len(scores_et)))

print("Mean of scores for gb:", sum(scores_gb) / float(len(scores_et)))
model_xg.fit(tweets_num_mod,target)

# plot feature importance for xgboost

plot_importance(model_xg)

plt.title('Feature importance', bbox={'facecolor':'0.8', 'pad':0})

plt.show()