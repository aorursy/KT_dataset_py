#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# import ploty for visualization
import plotly
import plotly.offline as py # make offline 
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')
import warnings
warnings.filterwarnings('ignore')
# this will allow ploting inside the notebook
%matplotlib inline
yelp_bdf = pd.read_csv('../input/yelp_business.csv')
yelp_attr = pd.read_csv('../input/yelp_business_attributes.csv')
yelp_bizhrs = pd.read_csv('../input/yelp_business_hours.csv')
cols_v = list(yelp_attr.columns.values)[1:]

for i in range(len(cols_v)):
    #print(cols_v[i])
    yelp_attr[cols_v[i]].replace('Na', np.nan, inplace=True)
    yelp_attr[cols_v[i]].replace('True', 1, inplace=True)
    yelp_attr[cols_v[i]].replace('False', 0, inplace=True)
yelp_rev = pd.read_csv('../input/yelp_review.csv',nrows = 10000)
yelp_rev.head(2)
yelp_user = pd.read_csv('../input/yelp_user.csv')
yelp_user[['compliment_profile', 'compliment_writer',
       'cool', 'elite', 'fans', 'friends', 'funny', 'name', 'review_count',
       'useful', 'user_id', 'yelping_since']].head(2)
yelp_user[yelp_user['friends'].notnull()].head(3)
yelp_tip = pd.read_csv('../input/yelp_tip.csv',nrows = 10000)
yelp_tip.head(2)
yelp_checkin = pd.read_csv('../input/yelp_checkin.csv')
yelp_checkin.head(2)
#yelp_rev['business_id'].isin(yelp_bdf['business_id']).value_counts()
#Merge review & business data on business_id. Get the business name, categories as well.
yelp_reviewd = pd.merge(yelp_rev, yelp_bdf, on='business_id', how='left', suffixes=('_review', '_biz'))
#yelp_reviewd.info()
#Merge review & tips data on business_id. 
yelp_reviewd1 = pd.merge(yelp_reviewd, yelp_tip, on='business_id', how='left', suffixes=('_review', '_tip'))
#Merge review & checkin data on business_id. 
yelp_reviewd2 = pd.merge(yelp_reviewd1, yelp_checkin, on='business_id', how='left')
yelp_businessdf = yelp_reviewd2.copy()
ambiencelist = yelp_attr.filter(like='Ambience').columns.tolist()
y = int(len(ambiencelist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        #print(i,j,ambiencelist[0])
        sns.countplot(yelp_attr[ambiencelist[0]], ax=ax[i,j], palette="Set1")
        del ambiencelist[0]
fig.tight_layout()        
bplist = yelp_attr.filter(like='BusinessParking').columns.tolist()
y = int(len(bplist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        #print(i,j,ambiencelist[0])
        sns.countplot(yelp_attr[bplist[0]], ax=ax[i,j], palette="Set1")
        del bplist[0]
fig.tight_layout()        

bnlist = yelp_attr.filter(like='BestNights').columns.tolist()
y = int(len(bnlist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        #print(i,j,ambiencelist[0])
        sns.countplot(yelp_attr[bnlist[0]], ax=ax[i,j], palette="Set1")
        del bnlist[0]
fig.tight_layout()        
meallist = yelp_attr.filter(like='GoodForMeal').columns.tolist()
y = int(len(meallist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        sns.countplot(yelp_attr[meallist[0]], ax=ax[i,j], palette="Set1")
        del meallist[0]
fig.tight_layout()        

dtlist = yelp_attr.filter(like='DietaryRestrictions').columns.tolist()
del dtlist[0]
y = int(len(dtlist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        sns.countplot(yelp_attr[dtlist[0]], ax=ax[i,j], palette="Set1")
        del dtlist[0]
fig.tight_layout()        
mlist = yelp_attr.filter(like='Music').columns.tolist()
y = int(len(mlist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        sns.countplot(yelp_attr[mlist[0]], ax=ax[i,j], palette="Set1")
        del mlist[0]
fig.tight_layout()        
hlist = yelp_attr.filter(like='HairSpecializesIn').columns.tolist()
y = int(len(hlist)/2)
fig, ax =plt.subplots(2, y, figsize=(8,6))
for i in range(2):
    for j in range(y):
        sns.countplot(yelp_attr[hlist[0]], ax=ax[i,j], palette="Set1")
        del hlist[0]
fig.tight_layout()        
alist00 = ['AgesAllowed', 'Alcohol', 'NoiseLevel', 'WiFi','RestaurantsAttire', 'Smoking']
alist = ['BikeParking', 
          'BusinessAcceptsCreditCards', 'Caters',
          'RestaurantsCounterService']
alist1 = [ 'DriveThru', 'CoatCheck', 'RestaurantsTableService','DogsAllowed', 
          'BYOB', 'BusinessAcceptsBitcoin']  
alist2 = [ 'RestaurantsDelivery','GoodForDancing',
          'RestaurantsGoodForGroups', 'RestaurantsReservations', 
           'RestaurantsTakeOut', 'ByAppointmentOnly']
alist3 = ['GoodForKids', 'HappyHour', 'HasTV', 
          'Open24Hours', 'OutdoorSeating', 'WheelchairAccessible']


fig, ax =plt.subplots(2, 2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        sns.countplot(yelp_attr[alist[0]], ax=ax[i,j], palette="Set1")
        del alist[0]
fig.tight_layout()      

fig, ax =plt.subplots(2, 3, figsize=(8,6))
for i in range(2):
    for j in range(3):
        sns.countplot(yelp_attr[alist00[0]], ax=ax[i,j], palette="Set1")
        del alist00[0]
fig.tight_layout() 
fig, ax =plt.subplots(2, 3, figsize=(8,6))
for i in range(2):
    for j in range(3):
        sns.countplot(yelp_attr[alist1[0]], ax=ax[i,j], palette="Set3")
        del alist1[0]
fig.tight_layout()    

fig, ax =plt.subplots(2, 3, figsize=(8,6))
for i in range(2):
    for j in range(3):
        sns.countplot(yelp_attr[alist3[0]], ax=ax[i,j], palette="Set2")
        del alist3[0]
fig.tight_layout() 
fig, ax =plt.subplots(2, 3, figsize=(8,6))
for i in range(2):
    for j in range(3):
        sns.countplot(yelp_attr[alist2[0]], ax=ax[i,j], palette="Set1")
        del alist2[0]
fig.tight_layout()   
#yelp_businessdf['categories'][yelp_businessdf['categories'].notnull()] = yelp_businessdf['categories'][yelp_businessdf['categories'].notnull()].apply(','.join)
yelp_businessdf['categories'].head()
cat_list = set()
for sstr in yelp_businessdf['categories'][yelp_businessdf['categories'].notnull()].str.split(';'):
    cat_list = set().union(sstr, cat_list)
cat_list = list(cat_list)
#cat_list.remove('')
cat_count = []
for cat in cat_list:
    cat_count.append([cat,yelp_businessdf['categories'].str.contains(cat).sum()])
names = ['cat_name','cat_count']
cat_df = pd.DataFrame(data=cat_count, columns=names)
cat_df.sort_values("cat_count", inplace=True, ascending=False)
cat_df.head(10)
plt.subplots(figsize=(8, 8))
labels=cat_df['cat_name'][cat_df['cat_count']>50000]
cat_df['cat_count'][cat_df['cat_count']>50000].plot.bar( align='center', alpha=0.5, color='red')
y_pos = np.arange(len(labels))
#plt.yticks(y_pos, labels)
plt.xticks(y_pos, labels)
plt.xlabel('Business Categories')
plt.ylabel('Categories Count')

plt.show()
plt.figure(figsize=(8,6))
ax = sns.countplot(yelp_businessdf['stars_biz'])
plt.title('Business Ratings');
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
ax = sns.countplot(yelp_businessdf['stars_review'])
plt.title('Review Ratings');
yelp_businessdf.name.value_counts().index[:20].tolist()
biz_cnt = pd.DataFrame(yelp_businessdf['name'].value_counts()[:20])
plt.figure(figsize=(12,6))
g = sns.barplot(x=biz_cnt.index, y=biz_cnt['name'], palette = 'Set1')
plt.title('List of most reviewed Businesses');
g.set_xticklabels(g.get_xticklabels(),rotation=90)
plt.show()
city_cnt = pd.DataFrame(yelp_businessdf['city'].value_counts()[:20])
state_cnt = pd.DataFrame(yelp_businessdf['state'].value_counts()[:20])
plt.figure(figsize=(12,6))
g = sns.barplot(x=city_cnt.index, y=city_cnt['city'], palette = 'Set1')
plt.title('List of most reviewed Cities');
g.set_xticklabels(g.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize=(12,6))
g = sns.barplot(x=state_cnt.index, y=state_cnt['state'], palette = 'Set1')
plt.title('List of most reviewed States');
g.set_xticklabels(g.get_xticklabels(),rotation=90)
plt.show()
from datetime import datetime
yelp_businessdf['date_review'] = pd.to_datetime(yelp_businessdf['date_review'])
yelp_businessdf['date_review'] = pd.to_datetime(yelp_businessdf['date_review'], format='%Y%m%d')
yelp_businessdf['month_review'] = yelp_businessdf.date_review.dt.to_period('M')
#Reviews per year and month

grp_date = yelp_businessdf.groupby(['date_review'])['business_id'].count()
grp_month = yelp_businessdf.groupby(['month_review'])['business_id'].count()

ts = pd.Series(grp_date)
ts.plot(kind='line', figsize=(20,10),title='Reviews per year')
plt.show()

ts = pd.Series(grp_month)
ts.plot(kind='line', figsize=(20,10),title='Reviews per month')
plt.show()
aggregations = {
    'review_id' : 'count',
    'cool':'sum',
    'funny':'sum',
    'useful':'sum',    
    'stars_review': 'mean'
}
#The user who has given most reviews
user_top10 = yelp_businessdf.groupby(['user_id_review'], as_index=False).agg(aggregations).sort_values(by='review_id', ascending=False)
user_top10.head(3)
#The user who has given most helpful reviews
yelp_businessdf.groupby(['user_id_review'], as_index=False).agg(aggregations).sort_values(by='useful', ascending=False).head(3)
from collections import Counter 
# wordcloud in python
from wordcloud import WordCloud, STOPWORDS 

import re 
import string
import nltk # preprocessing text
from textblob import TextBlob
i = nltk.corpus.stopwords.words('english')
# punctuations to remove
j = list(string.punctuation)
# finally let's combine all of these
stopwords = set(i).union(j).union(('thiswas','wasbad','thisis','wasgood','isbad','isgood','theres','there'))
# function for pre-processing the text of reviews: this function remove punctuation, stopwords and returns the list of words
def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())                  
    x = [w for w in x.split() if w not in set(stopwords)]  
    return ' '.join(x)
yelp_top_reviewd_biz = yelp_businessdf.loc[yelp_businessdf['name'].isin(biz_cnt.index)]
yelp_top_reviewd_biz['text_processed'] = yelp_top_reviewd_biz['text_review'].apply(preprocess)
wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=500, background_color='white',)
wordcloud.generate(str(set(yelp_top_reviewd_biz['text_processed'])))
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Top Reviewed words", fontsize=40,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()
yelp_top_reviewd_biz['tip_text_processed'] = yelp_top_reviewd_biz['text_tip'].dropna().apply(preprocess)
wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=500, background_color='white',)
wordcloud.generate(str(set(yelp_top_reviewd_biz['tip_text_processed'])))
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Top Reviewed words from Tips", fontsize=40,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()
def sentiment(x):
    sentiment = TextBlob(x)
    return sentiment.sentiment.polarity
yelp_top_reviewd_biz['text_sentiment'] = yelp_top_reviewd_biz['text_processed'].apply(sentiment)
yelp_top_reviewd_biz['sentiment'] = ''
yelp_top_reviewd_biz['sentiment'][yelp_top_reviewd_biz['text_sentiment'] > 0] = 'positive'
yelp_top_reviewd_biz['sentiment'][yelp_top_reviewd_biz['text_sentiment'] < 0] = 'negative'
yelp_top_reviewd_biz['sentiment'][yelp_top_reviewd_biz['text_sentiment'] == 0] = 'neutral'
plt.figure(figsize=(6,6))
ax = sns.countplot(yelp_top_reviewd_biz['sentiment'])
plt.title('Review Sentiments');
yelp_top_reviewd_biz_posr = pd.DataFrame(yelp_top_reviewd_biz['text_processed'][ yelp_top_reviewd_biz['sentiment'] == 'positive'])
yelp_top_reviewd_biz_negr = pd.DataFrame(yelp_top_reviewd_biz['text_processed'][ yelp_top_reviewd_biz['sentiment'] == 'negative'])
yelp_top_reviewd_biz_neutr = pd.DataFrame(yelp_top_reviewd_biz['text_processed'][ yelp_top_reviewd_biz['sentiment'] == 'neutral'])
wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=500, background_color='white',)
wordcloud.generate(str(set(yelp_top_reviewd_biz_posr['text_processed'])))
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Positive Sentiment", fontsize=40,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()
wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=500, background_color='white',)
wordcloud.generate(str(set(yelp_top_reviewd_biz_negr['text_processed'])))
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Negative Sentiment", fontsize=40,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()
wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=500, background_color='white',)
wordcloud.generate(str(set(yelp_top_reviewd_biz_neutr['text_processed'])))
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Neutral Sentiment", fontsize=40,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()
