# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import string
import json


import numpy as np 
import pandas as pd 
from time import time
import re
import string
import os
import emoji
import collections


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt



from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#import requests
#from mpl_toolkits.basemap import Basemap
#from geopy.geocoders import Nominatim

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/twitterData.xlsx')
df.head(5)
df.info()
#barplot function

def drawbarplot(x,y,xlabel,title,figsize=(10,10)):
    plt.figure(figsize=figsize)    
    sns.barplot(x=x,y=y,palette = 'terrain',orient='h',order=y)
    for i,v in enumerate(x):
        plt.text(0.8,i,v,color='k',fontsize=10)
    
    plt.title(title,fontsize=20)
    plt.xlabel(xlabel,fontsize =14)
    plt.show()


import datetime
#Converting datetime string to datetime
df['TweetPostedTime'] = [datetime.datetime.strptime(d, '%a %b %d %H:%M:%S %z %Y') for d in df['TweetPostedTime']]
df['UserSignupDate'] = [datetime.datetime.strptime(d, '%a %b %d %H:%M:%S %z %Y') for d in df['UserSignupDate']]
df['TweetPostedTime_hour'] = [d.hour for d in df['TweetPostedTime']]
count =  df['TweetPostedTime_hour'].value_counts()
drawbarplot(x=count.values,y=count.index,xlabel='count',title='Time of the day distribution',figsize=(10,10))

#Which user is most active on twitter
count=df['UserName'].value_counts()
df_count=pd.DataFrame()
df_count['Username'] = count.index
df_count['activeCount'] = count.values
df_count = df_count.iloc[:50,:]
drawbarplot(x=df_count.activeCount,y=df_count.Username,xlabel='count',title='Top 50 active user in given time span',figsize=(16,16))
df_tweetcount = df.loc[:,['UserName','UserTweetCount']]
df_tweetcount.sort_values(by='UserTweetCount',ascending=False,inplace=True)
df_tweetcount.drop_duplicates(subset='UserName',keep='first',inplace=True)
df_count=pd.DataFrame()
df_count = df_tweetcount.iloc[:50,:]
drawbarplot(x=df_count.UserTweetCount,y=df_count.UserName,xlabel='count',title='All time Top 50 active user',figsize=(16,16))
#Which user has most number of follower on twitter
df_userfollower = df.loc[:,['UserName','UserFollowersCount']]
df_userfollower.sort_values(by='UserFollowersCount',ascending=False,inplace=True)
df_userfollower.drop_duplicates(subset='UserName',keep='first',inplace=True)
df_count =df_userfollower.iloc[:50,:]
drawbarplot(x=df_count.UserFollowersCount,y=df_count.UserName,xlabel='count',title='Top 50 active user',figsize=(16,16))


#Which user has most number of Friend on twitter
df_userFriend = df.loc[:,['UserName','UserFriendsCount']]
df_userFriend.sort_values(by='UserFriendsCount',ascending=False,inplace=True)
df_userFriend.drop_duplicates(subset='UserName',keep='first',inplace=True)
df_count = df_userFriend.iloc[:50,:]
drawbarplot(x=df_count.UserFriendsCount,y=df_count.UserName,xlabel='count',title='Top 50 Friendly user',figsize=(16,16))

#function to extract @mentions and #tags
def extracter():
    mentions={}
    tags={}
    for i in df_trend_user.index:
        tokens = df_trend_user['TweetBody'][i].split()    
        for token in tokens:
            if('@' in token[0] and len(token) > 1):
                if token.strip('@') in mentions:
                    mentions[token.strip('@')] += 1
                else:
                    mentions[token.strip('@')] = 1
        
        
            if('#' in token[0] and len(token) > 1):
                if token.strip('#') in tags:
                    tags[token.strip('#')] += 1
                else:
                    tags[token.strip('#')] = 1    
                    
    return mentions,tags    
df_trend_user = df.loc[:,['UserName','TweetBody','TweetRetweetFlag','TweetRetweetCount','TweetFavoritesCount','TweetHashtags']]
#df_trend_user.shape
mentions ,tags = extracter()
mentions_keys = list(mentions.keys())
mentions_values = list(mentions.values())
tags_keys = list(tags.keys())
tags_values = list(tags.values())
df_mention = pd.DataFrame(columns=['mentions','m_count'])
df_mention['mentions'] = mentions_keys
df_mention['m_count'] = mentions_values
df_mention.sort_values(ascending=False,by='m_count',inplace=True)
df_count = df_mention.iloc[:50,:]
drawbarplot(x=df_count.m_count,y=df_count.mentions,xlabel='Count of mentions',title='Top 50 mentions',figsize=(16,16))
df_tags =pd.DataFrame(columns=['tags','t_count'])
df_tags['tags'] = tags_keys
df_tags['t_count'] = tags_values
df_tags.sort_values(ascending=False,by='t_count',inplace=True)
df_count = df_tags.iloc[:50,:]
drawbarplot(x=df_count.t_count,y=df_count.tags,xlabel='Count of tags',title='Top 50 Tags',figsize=(16,16))
df_trend_tweets = df.loc[:,['UserName','TweetBody','TweetRetweetFlag','TweetRetweetCount','TweetFavoritesCount','TweetHashtags']]
df_trend_tweets = df_trend_tweets.loc[(df_trend_tweets['TweetRetweetFlag'] == 1) & (df_trend_tweets['TweetRetweetCount'] > 1)]
df_trend_tweets.drop(columns='TweetRetweetFlag',axis=1,inplace=True)
#Removing duplicate tweets
df_trend_tweets.drop_duplicates(keep='first',subset='TweetBody',inplace=True)
df_trend_tweets['TweetBody']= df_trend_tweets['TweetBody'].str.lower()
#df_trend_user.shape
df_trend_tweets.head()
st_words = set(STOPWORDS)
#enhancing stopword by removing @mentions and shorthands
st_words.update(['https','CO','RT','Please','via','amp','place','new','ttot','best','great','top','ht','ysecrettravel','ysecrettravel_'])
st_words.update([s.lower() for s in mentions_keys])
wc = WordCloud(height=600,repeat=False,width=1400,max_words=1000,stopwords=st_words,colormap='terrain',background_color='Cyan',mode='RGBA').generate(' '.join(df_trend_tweets['TweetBody'].dropna().astype(str)))
plt.figure(figsize = (16,16))
plt.imshow(wc)
plt.title('Tweets Wordcloud')
plt.axis('off')
plt.show()
df_userfollower= df_userfollower.merge(df[['UserTweetCount']],left_index=True,right_index=True,how='left',sort=False)
df_count = df_userfollower.iloc[:50,:]
drawbarplot(x=df_count.UserTweetCount,y=df_count.UserName,xlabel='Total Tweet counts',title='Total Tweet counts by top 50 leaders in Twitter',figsize=(16,16))
df_userfollower.sort_values(by='UserTweetCount',ascending=False,inplace=True)
df_count = df_userfollower.iloc[:50,:]
drawbarplot(x=df_count.UserTweetCount,y=df_count.UserName,xlabel='Total Tweet counts',title='Top 50 Tweeters among leaders in Twitter',figsize=(16,16))
#df_user_signup = df['UserrName','UserScreenName','UserLocation','UserDesciption','UserWarning','UserSignupDate']
#df.columns[df.columns.str.startswith('User')]
df_user_signup=df.loc[:,df.columns[df.columns.str.startswith('User')]]
df_user_signup.sort_values(by='UserSignupDate',ascending=True,inplace=True)
df_user_signup.drop_duplicates(inplace=True,keep='first',subset='UserName')
df_count = df_user_signup.iloc[:50,:]
drawbarplot(x=df_count.UserFollowersCount,y=df_count.UserName,title='Number of Followers of top 50 Earliest Users',xlabel='No. of Users',figsize=(16,16))
#df_count = df_user_signup.iloc[:50,:]
drawbarplot(x=df_count.UserFriendsCount,y=df_count.UserName,title='Number of Friends of top 50 Earliest Users',xlabel='No. of Users',figsize=(16,16))
df_user_signup['year_of_signup']=[d.year for d in df_user_signup['UserSignupDate']]
df_user_signup['month_of_signup']=[d.month for d in df_user_signup['UserSignupDate']]
count = df_user_signup['year_of_signup'].value_counts()
drawbarplot(x=count.values,y=count.index,xlabel='count',title='Year with Maximum User SignUp',figsize=(10,10))
#df_user_signup.groupby('year_of_signup')['month_of_signup'].transform('count')
months = {1:'Jan',
 2:'Feb',
 3:'March',
 4:'April',
 5:'May',
 6:'June',
 7:'July',
 8:'Aug',
 9:'Sept',
 10:'Oct',
 11:'Nov',
 12:'Dec'
}
df_user_signup['month_of_signup'] =df_user_signup['month_of_signup'].map(months)
df_count = df_user_signup.loc[df_user_signup['year_of_signup']==2016,['UserName','month_of_signup']]
count = df_count['month_of_signup'].value_counts()
drawbarplot(x=count.values,y=count.index,xlabel='count',title='In Which month with Maximum User SignedUp',figsize=(10,10))
country = {'united states':'united states of america','new york city':'new york'}
replc_list = ['us','ny','india','uk','california','chicago','los angeles','london']
val_list=['united states of america','new york','india','united kingdom','california','chicago','los angeles','london']
#us_list =['united states','us','usa','new york','chicago','los angeles','california','seattle','las vegas','new york city']
#uk_list = ['uk','london','england']
#india_list = ['mumbai','new delhi']
# Removing reduntant cities.
df_user_signup['UserLocation'] =df['UserLocation'].str.lower()
df_user_signup['UserLocation'].replace(country,inplace=True)
for i in range(0,len(val_list)):    
    df_user_signup['UserLocation'].replace(to_replace=r'^.*'+replc_list[i]+'.*$',value=val_list[i],regex=True,inplace=True)
count=df_user_signup['UserLocation'].value_counts()
df_count = pd.DataFrame()
df_count['UserLocation']=count.index
df_count['loc_count']=count.values
df_count=df_count.iloc[:25,:]
drawbarplot(x=df_count.loc_count,y=df_count.UserLocation,xlabel='Location Count',title='Top 25 place with maximum Twitter Users',figsize=(16,16))
df_place = df.loc[:,['tweet.place']]
df_place.dropna(axis=0,how='any',inplace=True)
df_place.reset_index(drop=True,inplace=True)
df_place.shape
df_place['tweet.place'] = df_place['tweet.place'].apply(json.loads)
df_place= pd.DataFrame(df_place['tweet.place'].tolist())
df_count = pd.DataFrame()
df_count['country_code'] = df_place['country_code'].value_counts().index
df_count['tw_count'] = df_place['country_code'].value_counts().values
df_count = df_count.merge(df_place[['country_code','country']],how='inner',on='country_code',left_index=False,right_index=False)
df_count.drop_duplicates(inplace=True,subset='country_code')
df_count.reset_index(drop=True,inplace=True)
df_count.head()
drawbarplot(x=df_count.tw_count,y=df_count.country,figsize=(16,16),xlabel='tweet Count',title='Location with max tweets')

df.head()
#Drop irrelevant columns
colsToDrop=['TweetID','TweetSource','TweetPostedTime', 'TweetPlaceID','TweetPlaceAttributes',
       'TweetPlaceContainedWithin', 'UserID','UserDescription','UserLink', 'UserExpandedLink',
        'UserListedCount','tweet.place']
df_final=df.drop(colsToDrop,axis=1)
df_final.shape
df_final.isnull().mean()
##Dropping columns which are having more than 95%Null Values
nullCols = ['TweetInReplyToStatusID','TweetInReplyToUserID','TweetInReplyToScreenName','TweetPlaceName',
           'TweetPlaceFullName','TweetCountry','TweetPlaceBoundingBox']
df_final.drop(nullCols,axis=1,inplace=True)
#Baseestimator class to extract fetures from tweetbody

class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        #count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        #count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags                           
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df
tc = TextCounts()
df_feature =  tc.fit_transform(df_final['TweetBody'])
df_feature.head(10)
df_feature.shape
#Converting TweetRetweetFlag to integer
df_final['TweetRetweetFlag'] = df_final['TweetRetweetFlag'].map({True:1,False:0})
#Extracting features from date
df_final['year_of_signup']=[d.year for d in df['UserSignupDate']]
df_final['month_of_signup']=[d.month for d in df['UserSignupDate']]
df_final.drop(columns='UserSignupDate',inplace=True,axis=1)
#Cleaning Country attribute
country = {'united states':'united states of america','new york city':'new york'}
replc_list = ['us','ny','india','uk','california','chicago','los angeles','london']
val_list=['united states of america','new york','india','united kingdom','california','chicago','los angeles','london']

df_final['UserLocation'] =df_final['UserLocation'].str.lower()
df_final['UserLocation'].replace(country,inplace=True)
for i in range(0,len(val_list)):    
    df_final['UserLocation'].replace(to_replace=r'^.*'+replc_list[i]+'.*$',value=val_list[i],regex=True,inplace=True)
#df_final.head()
#df_final.shape
df_feature.shape
## Cleaning Tweet Body
class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        #stopwords_list = st_words
        stopwords_list=STOPWORDS
        # Some words which might indicate a certain sentiment are kept via a whitelist        
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)        
        return clean_X
ct = CleanText()
df_final['TweetBody'] = ct.fit_transform(df_final.TweetBody)
#Imputing '[no text]' value where there is no text
df_final.loc[df_final['TweetBody'] == '','TweetBody'] = '[no text]'
df_final.drop(columns='TweetHashtags',axis=1,inplace=True)
cv = CountVectorizer()
bow = cv.fit_transform(df_final['TweetBody'])
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(y="word", x="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax,orient='h')
plt.show();
df_final = pd.concat([df_final,df_feature],ignore_index=False,axis=1,)
#Encoding Text columns into numeric
df_final['UserName'] = pd.factorize(df_final['UserName'])[0]
df_final['UserScreenName'] = pd.factorize(df_final['UserScreenName'])[0]
df_final['UserLocation'] = pd.factorize(df_final['UserLocation'])[0]
df_final.skew()
from sklearn.model_selection import train_test_split
X=df_final.drop(['TweetRetweetCount'],axis=1)
y=df_final['TweetRetweetCount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22,shuffle=True)
print('X_train.shape %s, X_test.shape %s\ny_train.shape %s, y_test.shape %s'%(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
#Tokenizing text with scikit learn countVectorizer
countvec= CountVectorizer()
X_train_count = countvec.fit_transform(X_train.TweetBody)
X_test_count = countvec.transform(X_test.TweetBody)
print('X_train_count.shape %s\nX_test_count.shape %s'%(X_train_count.shape,X_test_count.shape))
#Moving from Occurance to frequency using Term Frequency and 
#downscale weights for words that occur in most tweet as tthey may carry less info than those which 
#occur more frequently usinf Tf-idf

from sklearn.feature_extraction.text import TfidfTransformer
tfid = TfidfTransformer(use_idf=True)
X_train_tfidf = tfid.fit_transform(X_train_count)
X_test_tfidf = tfid.transform(X_test_count)

print('X_train_tfidf.shape %s\nX_test_tfidf.shape %s'%(X_train_tfidf.shape,X_test_tfidf.shape))
### Combining all the features
from scipy import sparse
numCols = X_train.columns
numCols =numCols.drop('TweetBody')

X_train_num_feature = X_train[numCols].values
X_test_num_feature = X_test[numCols].values

X_traindata = sparse.hstack((X_train_tfidf,X_train_num_feature))
X_testdata = sparse.hstack((X_test_tfidf,X_test_num_feature))

print('X_traindata.shape %s\nX_testdata.shape %s'%(X_traindata.shape,X_testdata.shape))
from sklearn.decomposition import TruncatedSVD

n_components = 1000
pca = TruncatedSVD(n_components)
X_traindata = pca.fit_transform(X_traindata)
X_testdata = pca.transform(X_testdata)
type(X_traindata)
### Normalise feature
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler(with_mean=False)
#scaled_X_train = scaler.fit_transform(X_traindata)
#scaled_X_test = scaler.transform(X_testdata)
lr = LinearRegression().fit(X_traindata,y_train)
y_pred = lr.predict(X_testdata)
#y_pred_train = lr.predict(X_traindata)
# r2_score score: 1 is perfect prediction
#lr_score=r2_score(y_pred=y_pred_train,y_true=y_train)
#print("Variance score (r2_score): %f"%lr_score)
#print('Model accuracy:%.2f '%(lr_score*100))
#print("Root mean squared error of test:%f"%sqrt(mean_squared_error(y_train,y_pred_train)))
#Applying LR on CV set and checking it's accuracy.
# r2_score score: 1 is perfect prediction
lr_score=r2_score(y_pred=y_pred,y_true=y_test)
print("Variance score (r2_score): %f"%lr_score)
print('Model accuracy:%.2f '%(lr_score*100))
print("Root mean squared error of test:%f"%sqrt(mean_squared_error(y_test,y_pred)))
plt.spy(X_test_tfidf)
plt.show()
