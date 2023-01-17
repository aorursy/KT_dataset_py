# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head(5)
train.shape
#the train dataset is reasonably balanced in that there are a roughly similar proportion of 1 and 0 in target column

train.target.sum() / train.shape[0]
#standard practice is to concat train and test as test may contain text never seen before 

df = pd.concat([train,test],axis=0)
#settings that you use for count vectorizer will go here

tfidf_vectorizer=TfidfVectorizer(use_idf=True,

                                ngram_range=(1,5),

                                #token_pattern='[a-zA-Z]{3}',#only alpha data >= 3 in length

                                stop_words='english',

                                min_df=3,

                                max_features=5000,

                                binary=False # try a separate one with this set to true

                                )



#fit transform

tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(df['text'])
#tfidf_vectorizer_vectors


df['hashtags'] = df['text'].str.findall(r'#.*?(?=\s|$)').apply(lambda x: ','.join(map(str, x)))

train['hashtags'] = train['text'].str.findall(r'#.*?(?=\s|$)').apply(lambda x: ','.join(map(str, x))) 

test['hashtags'] = test['text'].str.findall(r'#.*?(?=\s|$)').apply(lambda x: ','.join(map(str, x))) 

#length of tweet

train['len'] = train['text'].str.len()

test['len'] = test['text'].str.len()
#number of mentions @

train['no_ats_spec_char'] = train['text'].str.count('@')

test['no_ats_spec_char'] = test['text'].str.count('@')
#number of hashtags # present 

train['no_hashtags_spec_char'] = train['text'].str.count('#')

test['no_hashtags_spec_char'] = test['text'].str.count('#')
#special characters total count

pat = '[(:/,#%\=@)]'

train['no_total_spec_char'] = train['text'].str.count(pat)

test['no_total_spec_char'] = test['text'].str.count(pat)
#position of mention @ in text

train['pos_at_char'] = train['text'].str.find('@') 

test['pos_at_char'] = test['text'].str.find('@') 
#position of hashtag # in text

train['pos_hashtag_char'] = train['text'].str.find('#') 

test['pos_hashtag_char'] = test['text'].str.find('#') 
#number of words

train['no_words'] = train['text'].str.count(' ').add(1).value_counts(sort=False)

test['no_words'] = test['text'].str.count(' ').add(1).value_counts(sort=False)
#average length of words

train['avg_len'] = train['len'] / train['no_words']

test['avg_len'] = test['len'] / test['no_words']
#categorise keyword data

#nb we should ensure that we apply same categories to train and test 

train['keyword'] = train['keyword'].replace(np.nan, 'None')

test['keyword'] = test['keyword'].replace(np.nan, 'None')

keyword_unique = pd.concat([train['keyword'],test['keyword']],axis=0).unique()

train['keyword'] = pd.Categorical(train['keyword'],categories=keyword_unique)

test['keyword'] = pd.Categorical(test['keyword'],categories=keyword_unique)

keyword_unique
#categorise location data

#nb we should ensure that we apply same categories to train and test 

train['location'] = train['location'].replace(np.nan, 'None')

test['location'] = test['location'].replace(np.nan, 'None')

location_unique = pd.concat([train['location'],test['location']],axis=0).unique()

train['location'] = pd.Categorical(train['location'],categories=location_unique)

test['location'] = pd.Categorical(test['location'],categories=location_unique)

#tweet contains numeric data

pat = '[01232456789]'

train['contains_number'] = np.where(train['text'].str.count(pat)>=1,1,0)

test['contains_number'] = np.where(test['text'].str.count(pat)>=1,1,0)
df.head()
#spit into train validation datasets

X_train, X_val, y_train, y_val = train_test_split(train,

                                                  train['target'],

                                                 random_state = 888,

                                                 test_size = 0.2,

                                                 shuffle = True)
#apply the tfidf to datasets

train_set = tfidf_vectorizer.transform(X_train['text'])

val_set = tfidf_vectorizer.transform(X_val['text'])

test_set = tfidf_vectorizer.transform(test['text'])
#set the parameters of the model

params = { "objective" : "binary", # binary classification is the type of business case we are running

        "metric" :"F1", #F1 score is 2 * (TP) / (TP + FP) is a standard metric to use

        "learning_rate" : 0.05, #the pace at which the model is allowed to reach it's objective of minimising the rsme.

        'num_iterations' : 2000,

        'num_leaves': 50, # minimum number of leaves in each boosting round

        "early_stopping": 50, #if the model does not improve after this many consecutive rounds, call a halt to training

        "max_bin": 200,

        "seed":888

    

    

}
#Run the model

m1_lgb = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1#, 

#                             learning_rate=0.05, 

#                             max_depth=20, 

#                             num_leaves=50, 

#                             n_estimators=1000, 

#                             max_bin=200

                           ) #params,  verbose_eval = 50)
m1_lgb.fit(train_set, y_train) 
#plot feature importance

feature_imp = pd.DataFrame({'Value':m1_lgb.feature_importances_,'Feature':tfidf_vectorizer.get_feature_names()})

plt.figure(figsize=(20, 10))

sns.set(font_scale = 1)

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 

                                                    ascending=False)[0:40])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances-01.png')

plt.show()
#generate predictions use predict_proba for ensemble models later on

pred_test = m1_lgb.predict_proba(test_set)[:,1]

pred_val = m1_lgb.predict_proba(val_set)[:,1]
#pred_val
#test.head()
#f1 score, between 0 and 1, the higher the better

f1_score(X_val['target'],np.where(pred_val>=0.5,1,0))
submission_m1 = test[['id']].copy()

submission_m1['target'] =  np.where(pred_test>= 0.5,1,0)
submission_m1.head()
submission_m1.shape
#save submission

submission_m1.to_csv('./submission_m1.csv',index=False)
#settings that you use for count vectorizer will go here

tfidf_vectorizer_hashtags=TfidfVectorizer(use_idf=True,

                                ngram_range=(1,1),

                                #token_pattern='[a-zA-Z]{3}',#only alpha data >= 3 in length

                                stop_words='english',

                                min_df=3,

                                max_features=500,

                                binary=False # try a separate one with this set to true

                                )
#fit transform

tfidf_vectorizer_vectors_hashtags=tfidf_vectorizer_hashtags.fit_transform(df['hashtags'])
train_set_hashtags = tfidf_vectorizer_hashtags.transform(X_train['hashtags'])

val_set_hashtags = tfidf_vectorizer_hashtags.transform(X_val['hashtags'])

test_set_hashtags = tfidf_vectorizer_hashtags.transform(test['hashtags'])
#Run the model

m1_lgb_hashtags = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1#, 

#                             learning_rate=0.05, 

#                             max_depth=20, 

#                             num_leaves=50, 

#                             n_estimators=1000, 

#                             max_bin=200

                                    ) #params,  verbose_eval = 50)
m1_lgb_hashtags.fit(train_set_hashtags, y_train) 
len(tfidf_vectorizer_hashtags.get_feature_names())
#tfidf_vectorizer_hashtags.get_feature_names()
m1_lgb_hashtags.feature_importances_.shape
#plot feature importance

feature_imp_hashtags = pd.DataFrame({'Value':m1_lgb_hashtags.feature_importances_,'Feature':tfidf_vectorizer_hashtags.get_feature_names()})

plt.figure(figsize=(20, 10))

sns.set(font_scale = 1)

sns.barplot(x="Value", y="Feature", data=feature_imp_hashtags.sort_values(by="Value", 

                                                    ascending=False)[0:40])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances-01.png')

plt.show()
pred_test_hashtags = m1_lgb_hashtags.predict_proba(test_set_hashtags)[:,1]

pred_val_hashtags = m1_lgb_hashtags.predict_proba(val_set_hashtags)[:,1]
f1_score(X_val['target'],np.where(pred_val_hashtags>=0.5,1,0))
submission_m2 = test[['id']].copy()

submission_m2['target'] =  np.where(pred_test_hashtags>= 0.5,1,0)
submission_m2.to_csv('./submission_m2.csv',index=False)
df.head()
import nltk

import string

from nltk.stem.porter import PorterStemmer



def tokenize(text):

    tokens = nltk.word_tokenize(text)

    stems = []

    for item in tokens:

        stems.append(PorterStemmer().stem(item))

    return stems
tfidf_stem = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
#fit transform

tfidf_vectorizer_vectors_stem=tfidf_stem.fit_transform(df['text'])
train_set_stem = tfidf_stem.transform(X_train['text'])

val_set_stem = tfidf_stem.transform(X_val['text'])

test_set_stem = tfidf_stem.transform(test['text'])
#Run the model

m1_lgb_stem = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1, 

                            learning_rate=0.05, 

                            max_depth=20, 

                            num_leaves=50, 

                            n_estimators=1000, 

                            max_bin=200) #params,  verbose_eval = 50)
m1_lgb_stem.fit(train_set_stem, y_train) 
#plot feature importance

feature_imp_stem = pd.DataFrame({'Value':m1_lgb_stem.feature_importances_,'Feature':tfidf_stem.get_feature_names()})

plt.figure(figsize=(20, 10))

sns.set(font_scale = 1)

sns.barplot(x="Value", y="Feature", data=feature_imp_stem.sort_values(by="Value", 

                                                    ascending=False)[0:40])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances-01.png')

plt.show()
pred_test_stem = m1_lgb_stem.predict_proba(test_set_stem)[:,1]

pred_val_stem = m1_lgb_stem.predict_proba(val_set_stem)[:,1]

f1_score(X_val['target'],np.where(pred_val_stem>=0.5,1,0))
submission_m3 = test[['id']].copy()

submission_m3['target'] =  np.where(pred_test_stem>= 0.5,1,0)
submission_m3.to_csv('./submission_m3.csv',index=False)
df.columns
train_set_other = X_train[['len','no_ats_spec_char', 'no_hashtags_spec_char', 'no_total_spec_char','pos_at_char', 'pos_hashtag_char', 'contains_number','avg_len','keyword','location']]

val_set_other = X_val[['len','no_ats_spec_char', 'no_hashtags_spec_char', 'no_total_spec_char','pos_at_char', 'pos_hashtag_char', 'contains_number','avg_len','keyword','location']]

test_set_other = test[['len','no_ats_spec_char', 'no_hashtags_spec_char', 'no_total_spec_char','pos_at_char', 'pos_hashtag_char', 'contains_number','avg_len','keyword','location']]
#set the parameters

params = { "objective" : "binary", # binary classification is the type of business case we are running

        "metric" :"F1", #F1 score is 2 * (TP) / (TP + FP) is a standard metric to use

        "learning_rate" : 0.05, #the pace at which the model is allowed to reach it's objective of minimising the rsme.

        'num_iterations' : 500,

        'num_leaves': 50, # minimum number of leaves in each boosting round

        "early_stopping": 50, #if the model does not improve after this many consecutive rounds, call a halt to training

        "max_bin": 200,

        "seed":888

    

    

}


#Run the model

m5_lgb = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1#, 

                            #learning_rate=0.05, 

                            #max_depth=20, 

                            #num_leaves=50, 

                            #n_estimators=1000, 

                            #max_bin=200

                           ) #params,  verbose_eval = 50)

m5_lgb
m5_lgb.fit(train_set_other, y_train) 


#plot feature importance

feature_imp = pd.DataFrame({'Value':m5_lgb.feature_importances_,'Feature':train_set_other.columns})

plt.figure(figsize=(20, 10))

sns.set(font_scale = 1)

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 

                                                    ascending=False)[0:40])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances-01.png')

plt.show()

pred_test_other = m5_lgb.predict_proba(test_set_other)[:,1]

pred_val_other = m5_lgb.predict_proba(val_set_other)[:,1]
f1_score(X_val['target'],np.where(pred_val_other>=0.5,1,0))


submission_m5 = test[['id']].copy()

submission_m5['target'] =  np.where(pred_test_other>= 0.5,1,0)
submission_m5.to_csv('./submission_m5.csv',index=False)
from tqdm import tqdm

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus=create_corpus(df)
corpus[:5]
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.twitter.27B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
embedding_dict
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
#file we train on:

tweet_pad




word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))



word_index
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec

            
train_glove=tweet_pad[:train.shape[0]]

test_glove=tweet_pad[train.shape[0]:]
X_train_glove,X_val_glove,y_train_glove,y_val_glove=train_test_split(train_glove,train['target'].values,test_size=0.20)

print('Shape of train',X_train_glove.shape)

print("Shape of Validation ",X_val_glove.shape)




#Run the model

m6_lgb = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1, 

                            learning_rate=0.05, 

                            max_depth=20, 

                            num_leaves=50, 

                            n_estimators=1000, 

                            max_bin=200) #params,  verbose_eval = 50)
m6_lgb.fit(X_train_glove, y_train_glove) 
pred_test_glove = m6_lgb.predict_proba(test_glove)[:,1]

pred_val_glove = m6_lgb.predict_proba(X_val_glove)[:,1]
f1_score(X_val['target'],np.where(pred_val_glove>=0.5,1,0))


submission_m6 = test[['id']].copy()

submission_m6['target'] =  np.where(pred_test_glove>= 0.5,1,0)


submission_m6.to_csv('./submission_m6.csv',index=False)
#settings that you use for count vectorizer will go here

tfidf_vectorizer_binary=TfidfVectorizer(use_idf=True,

                                ngram_range=(1,5),

                                token_pattern='[a-zA-Z]{2,20}',#only alpha data >= 2 in length

                                #stop_words='english',

                                min_df=3,

                                max_df=0.70,

                                max_features=5000,

                                binary=True # try a separate one with this set to true

                                )



#fit transform

tfidf_vectorizer_vectors_binary=tfidf_vectorizer_binary.fit_transform(df['text'])
train_set_binary = tfidf_vectorizer_binary.transform(X_train['text'])

val_set_binary = tfidf_vectorizer_binary.transform(X_val['text'])

test_set_binary = tfidf_vectorizer_binary.transform(test['text'])
#Run the model

m7_lgb_binary = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1, 

                            learning_rate=0.05, 

                            max_depth=20, 

                            num_leaves=50, 

                            n_estimators=1000, 

                            max_bin=200) #params,  verbose_eval = 50)

m7_lgb_binary.fit(train_set_binary, y_train) 
#plot feature importance

feature_imp_stem = pd.DataFrame({'Value':m7_lgb_binary.feature_importances_,'Feature':tfidf_vectorizer_binary.get_feature_names()})

plt.figure(figsize=(20, 10))

sns.set(font_scale = 1)

sns.barplot(x="Value", y="Feature", data=feature_imp_stem.sort_values(by="Value", 

                                                    ascending=False)[0:40])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances-01.png')

plt.show()
pred_test_binary = m7_lgb_binary.predict_proba(test_set_binary)[:,1]

pred_val_binary = m7_lgb_binary.predict_proba(val_set_binary)[:,1]
f1_score(X_val['target'],np.where(pred_val_binary>=0.5,1,0))


submission_m7 = test[['id']].copy()

submission_m7['target'] =  np.where(pred_test_binary>= 0.5,1,0)
submission_m7.to_csv('./submission_m7.csv',index=False)
texts = []

for tweet in X_train['text']: 

    texts.append(tweet.split())
texts1 = []

for tweet in texts:

    texts1.append([x for x in tweet if x not in stopwords.words()])
texts1[:2]
from gensim.corpora import Dictionary

# you can use any corpus, this is just illustratory



dictionary = Dictionary(texts1)

corpus = [dictionary.doc2bow(text) for text in texts1]



import numpy

numpy.random.seed(1) # setting random seed to get the same results each time.



from gensim.models import ldamodel

model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20, minimum_probability=1e-8)

model.show_topics()
# we can now get the LDA topic distributions for these

# get the disaster text words

bow_keyword_unique = model.id2word.doc2bow(keyword_unique)

lda_bow_disaster = model[bow_keyword_unique]
X_train.head()
from gensim.matutils import hellinger
X_train_lda = []

for tweet in texts1:

    bow_tweet = model.id2word.doc2bow(tweet) 

    lda_bow_tweet = model[bow_tweet]

    X_train_lda.append(hellinger(lda_bow_disaster, lda_bow_tweet))



X_train_lda
#Run the model

m8_lgb_lda = lgb.LGBMClassifier(objective='binary', 

                            metric='F1',

                            verbose=-1, 

                            learning_rate=0.05, 

                            max_depth=20, 

                            num_leaves=50, 

                            n_estimators=1000, 

                            max_bin=200) #params,  verbose_eval = 50)



m8_lgb_lda.fit(pd.DataFrame(X_train_lda,columns=['lda']), y_train)
texts_val = []

for tweet in X_val['text']: 

    texts_val.append(tweet.split())





texts_val1 = []

for tweet in texts_val:

    texts_val1.append([x for x in tweet if x not in stopwords.words()])



X_val_lda = []

for tweet in texts_val1:

    bow_tweet = model.id2word.doc2bow(tweet) 

    lda_bow_tweet = model[bow_tweet]

    X_val_lda.append(hellinger(lda_bow_disaster, lda_bow_tweet))
texts_test = []

for tweet in test['text']: 

    texts_test.append(tweet.split())





texts_test1 = []

for tweet in texts_test:

    texts_test1.append([x for x in tweet if x not in stopwords.words()])



X_test_lda = []

for tweet in texts_test1:

    bow_tweet = model.id2word.doc2bow(tweet) 

    lda_bow_tweet = model[bow_tweet]

    X_test_lda.append(hellinger(lda_bow_disaster, lda_bow_tweet))
pred_test_lda = m8_lgb_lda.predict_proba(pd.DataFrame(X_test_lda,columns=['lda']))[:,1]

pred_val_lda = m8_lgb_lda.predict_proba(pd.DataFrame(X_val_lda,columns=['lda']))[:,1]
f1_score(X_val['target'],np.where(pred_val_lda>=0.5,1,0))
submission_ens1 = test[['id']].copy()
submission_ens1['target'] =  (pred_test * 0.225 + 

                              pred_test_hashtags * 0.0 + 

                              pred_test_stem * 0.225 + 

                              pred_test_other * 0.15 + 

                              pred_test_glove * 0.09 + 

                              pred_test_binary * 0.3 +

                              pred_test_lda * 0.01) 
submission_ens1['target'] =  np.where(submission_ens1['target'] >= 0.5,1,0)
submission_ens1.to_csv('./submission_ens1.csv',index=False)
submission_ens1.head()