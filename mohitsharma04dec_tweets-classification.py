# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv')

df_test=pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv')

df_train.head()
#def preprocessing(text):

#    return text.lower()

#df_train['text']=df_train['text'].apply(preprocessing)

#df_test['text']=df_test['text'].apply(preprocessing)

#df_train.drop_duplicates()

print(df_train.shape,df_test.shape)
import seaborn as sns

sns.countplot(df_train.target)

df_train.head()
import matplotlib.pyplot as plt

kw_d=df_train[df_train.target==1].keyword.value_counts().head(10)

kw_nd=df_train[df_train.target==0].keyword.value_counts().head(10)

plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(kw_d, kw_d.index, color='c')

plt.title('Top keywords for disaster tweets')

plt.subplot(122)

sns.barplot(kw_nd, kw_nd.index, color='y')

plt.title('Top keywords for non-disaster tweets')

plt.show()
location_count=df_train.location.value_counts().head(15).index

print(location_count)

plt.figure(figsize=(9,5))

sns.countplot(y=df_train.location,order=location_count)

plt.title('Top location for disaster tweets')
raw_loc = df_train.location.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

top_only = df_train[df_train.location.isin(top_loc)]

top_l = top_only.groupby('location').mean()['target'].sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=top_l.index, y=top_l)

plt.axhline(np.mean(df_train.target))

plt.xticks(rotation=80)

plt.show()

from wordcloud import STOPWORDS

def clean_text(text):

    text=text.strip()

    text=re.sub(r'https?://\S+','',text)

    text=re.sub(r'\s\s+',' ',text)

    return text

    

def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or ''



def find_mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or ''



def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or ''



def process_text(df):

    

    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))

    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))

    df['links'] = df['text'].apply(lambda x: find_links(x))

    # Tweet length

    df['text_len'] = df['text_clean'].apply(len)

    # Word count

    df['word_count'] = df["text_clean"].apply(lambda x: len(str(x).split()))

    # Stopword count

    df['stop_word_count'] = df['text_clean'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    # Punctuation count

    #df['punctuation_count'] = df['text_clean'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # Count of hashtags (#)

    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split()))

    # Count of mentions (@)

    df['mention_count'] = df['mentions'].apply(lambda x: len(str(x).split()))

    # Count of links

    df['link_count'] = df['links'].apply(lambda x: len(str(x).split()))

    # Count of uppercase letters

    df['caps_count'] = df['text_clean'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))

    # Ratio of uppercase letters

    df['caps_ratio'] = df['caps_count'] / df['text_len']

    return df

    

train = process_text(df_train)

test = process_text(df_test)

print(df_train.shape,df_test.shape)
df_train.corr()['target'].drop('target').sort_values()
from sklearn.feature_extraction.text import CountVectorizer



count_vec_hash=CountVectorizer(analyzer='word',min_df=5)

hashtag_feat_train=count_vec_hash.fit_transform(df_train.hashtags)

hashtag_feat_test=count_vec_hash.transform(df_test.hashtags)

X_train_hash = pd.DataFrame(hashtag_feat_train.toarray(), columns=count_vec_hash.get_feature_names())

X_test_hash = pd.DataFrame(hashtag_feat_test.toarray(), columns=count_vec_hash.get_feature_names())



count_vec_mentions=CountVectorizer(analyzer='word',min_df=5)

mentions_feat_train=count_vec_mentions.fit_transform(df_train.mentions)

mentions_feat_test=count_vec_mentions.transform(df_test.mentions)

X_train_mentions = pd.DataFrame(mentions_feat_train.toarray(), columns=count_vec_mentions.get_feature_names())

X_test_mentions = pd.DataFrame(mentions_feat_test.toarray(), columns=count_vec_mentions.get_feature_names())



count_vec_link=CountVectorizer(min_df=5,token_pattern='https?://\S+')

link_feat_train=count_vec_link.fit_transform(df_train.links)

link_feat_test=count_vec_link.transform(df_test.links)

X_train_link = pd.DataFrame(link_feat_train.toarray(), columns=count_vec_link.get_feature_names())

X_test_link = pd.DataFrame(link_feat_test.toarray(), columns=count_vec_link.get_feature_names())







print(X_train_hash.shape,X_train_mentions.shape,X_train_link.shape)
_=(X_train_link.transpose().dot(df_train.target)/X_train_link.sum(axis=0)).sort_values(ascending=False)

plt.figure(figsize=(9,3))

sns.barplot(x=_,y=_.index)

plt.axvline(np.mean(train.target))

plt.title('% of disaster tweet given links')

plt.show()

hash_rank = (X_train_hash.transpose().dot(train['target']) / X_train_hash.sum(axis=0)).sort_values(ascending=False)

print('Hashtags with which 100% of Tweets are disasters: ')

print(list(hash_rank[hash_rank==1].index))

print('Total: ' + str(len(hash_rank[hash_rank==1])))

print('Hashtags with which 0% of Tweets are disasters: ')

print(list(hash_rank[hash_rank==0].index))

print('Total: ' + str(len(hash_rank[hash_rank==0])))

_=(X_train_mentions.transpose().dot(df_train.target)/X_train_mentions.sum(axis=0)).sort_values(ascending=False)

plt.figure(figsize=(15,6))

sns.barplot(x=_,y=_.index)

plt.axvline(np.mean(train.target))

plt.title('% of disaster tweet given links')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vec=TfidfVectorizer(analyzer='word',smooth_idf=True,lowercase=True,min_df=10,stop_words='english',ngram_range=(1,2))

tfidf_train=tfidf_vec.fit_transform(df_train['text_clean'])

tfidf_test=tfidf_vec.transform(df_test['text_clean'])

X_train_text = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vec.get_feature_names())

X_test_text = pd.DataFrame(tfidf_test.toarray(), columns=tfidf_vec.get_feature_names())

print(X_train_text.shape,X_test_text.shape)
df_train = df_train.join(X_train_link, rsuffix='_link')

df_train = df_train.join(X_train_mentions, rsuffix='_mention')

df_train = df_train.join(X_train_hash, rsuffix='_hashtag')

df_train = df_train.join(X_train_text, rsuffix='_text')

df_test = df_test.join(X_test_link, rsuffix='_link')

df_test = df_test.join(X_test_mentions, rsuffix='_mention')

df_test = df_test.join(X_test_hash, rsuffix='_hashtag')

df_test = df_test.join(X_test_text, rsuffix='_text')

print(df_train.shape,df_test.shape)
import category_encoders as ce



# Target encoding 

enco = ce.TargetEncoder(cols='keyword')

enco.fit(train['keyword'],train['target'])

df_train = df_train.join(enco.transform(df_train['keyword']).add_suffix('_target'))

df_test = df_test.join(enco.transform(df_test['keyword']).add_suffix('_target'))





# Target encoding

encoder = ce.TargetEncoder(cols='location')

encoder.fit(train['location'],train['target'])

df_train = df_train.join(encoder.transform(df_train['location']).add_suffix('_target'))

df_test = df_test.join(encoder.transform(df_test['location']).add_suffix('_target'))

features_to_drop = ['id', 'keyword','location','text','location','text_clean', 'hashtags', 'mentions','links']

#scaler = MinMaxScaler()



X_train = df_train.drop(columns = features_to_drop + ['target'])

X_test = df_test.drop(columns = features_to_drop)

y_train = df_train.target

print(df_train.shape,df_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 



lr=LogisticRegression()

pipeline = Pipeline([('scale',scaler), ('lr', lr),])

pipeline.fit(X_train, y_train)

y_test = pipeline.predict(X_test)
print ('Training accuracy: %.4f' % pipeline.score(X_train, y_train))

from sklearn.metrics import f1_score

print ('Training f-1 score: %.4f' % f1_score(y_train, pipeline.predict(X_train)))
from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y_train, pipeline.predict(X_train)))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)

cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

print('Cross validation F-1 score: %.3f' %np.mean(cv_score))
# Submit fine-tuned model

#y_test = pipeline.predict(X_test)

#y_test2 = pipeline_grid.predict(X_test2)

sub_sample=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submit = sub_sample.copy()

submit.target = pipeline.predict(X_test)

submit.to_csv('submit_lr.csv',index=False)