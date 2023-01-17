# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')



df
df.columns
df.dtypes
df.describe()
df.isna().sum()
df['user_description'] = df['user_description'].fillna('unknown')



df['hashtags'] = df['hashtags'].fillna('none')



df['source'] = df['source'].fillna('none')



df = df.dropna()



df.isna().sum()
df.isna().sum()
df['user_location'].isna().sum()
verified_df = df.query('user_verified == True').reset_index(drop=True)



unverified_df = df.query('user_verified == False').reset_index(drop=True)
top_verified_df = verified_df.loc[:,['user_name','user_location','user_followers','text']]



top_unverified_df = unverified_df.loc[:,['user_name','user_location','user_followers','text']]
import plotly.express as px



tweet_ct_ver = top_verified_df.groupby('user_name')['user_location'].count().reset_index()



tweet_ct_ver.columns = ['user','count']



tweet_ct_ver = tweet_ct_ver.sort_values(['count'])



fig = px.bar(tweet_ct_ver.tail(20), x='count',y='user',orientation='h')



fig.show()
import plotly.express as px



tweet_ct_unver = top_unverified_df.groupby('user_name')['user_location'].count().reset_index()



tweet_ct_unver.columns = ['user','count']



tweet_ct_unver = tweet_ct_unver.sort_values(['count'])



fig = px.bar(tweet_ct_unver.tail(20), x='count',y='user',orientation='h')



fig.show()
veri_loc_count = top_verified_df.groupby('user_location')['user_name'].count().reset_index()



veri_loc_count.columns = ['location','count']



veri_loc_count = veri_loc_count.sort_values(['count'])



fig = px.bar(veri_loc_count.tail(20), x='count',y='location',orientation='h')



fig.show()
unveri_loc_count = top_unverified_df.groupby('user_location')['user_name'].count().reset_index()



unveri_loc_count.columns = ['location','count']



unveri_loc_count = unveri_loc_count.sort_values(['count'])



fig = px.bar(unveri_loc_count.tail(20), x='count',y='location',orientation='h')



fig.show()
top_text_ver = top_verified_df.loc[:,['user_name','text']]



top_text_unver = top_unverified_df.loc[:,['user_name','text']]
top_followed_ver = top_verified_df.loc[:,['user_name','user_followers']]



top_followed_ver = top_followed_ver.groupby('user_name')['user_followers'].max().reset_index()



top_followed_unver = top_unverified_df.loc[:,['user_name','user_followers']]



top_followed_unver = top_followed_unver.groupby('user_name')['user_followers'].max().reset_index()





top_ver_followed_all = top_followed_ver.merge(top_text_ver,on='user_name')

top_unver_followed_all = top_followed_unver.merge(top_text_unver,on='user_name')



top_followed = top_followed.sort_values(['user_followers'])



fig = px.bar(top_followed.tail(20), x='user_followers',y='user_name',orientation='h')



fig.show()
ver_text_data = top_ver_followed_all['text']



ver_text_data = ver_text_data.str.replace("[^\w\s]","").str.lower()



unver_text_data = top_unver_followed_all['text']



unver_text_data = unver_text_data.str.replace("[^\w\s]","").str.lower()

import sklearn



from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words2 = get_top_n_bigram(ver_text_data, 100)

for word, freq in common_words2:

    print(word, freq)
topics_ver = pd.DataFrame(common_words2, columns=['topic','count']).sort_values(['count'])



fig = px.bar(topics_ver.tail(20), x='count',y='topic',width=1000, height=1000)



fig.show()
def get_top_n_unigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(1,1), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words3 = get_top_n_unigram(ver_text_data, 100)

for word, freq in common_words3:

    print(word, freq)
topics_ver2 = pd.DataFrame(common_words3, columns=['topic','count']).sort_values(['count'])



topics_ver2.drop(topics_ver2[topics_ver2['topic'] == 'amp'].index,inplace=True)



fig = px.bar(topics_ver2.tail(20), x='count',y='topic',width=1000, height=1000,orientation = 'h')



fig.show()
def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words4 = get_top_n_bigram(unver_text_data, 100)

for word, freq in common_words4:

    print(word, freq)
topics_unver = pd.DataFrame(common_words4, columns=['topic','count']).sort_values(['count'])



topics_unver.drop(topics_unver[topics_unver['topic'] == 'amp'].index,inplace=True)



fig = px.bar(topics_unver.tail(20), x='count',y='topic',width=1000, height=1000,orientation = 'h')



fig.show()