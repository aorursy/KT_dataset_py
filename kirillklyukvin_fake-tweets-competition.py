import numpy as np
from numpy import savetxt 
import pandas as pd 
import re
import gc
import random
import os
import tensorflow as tf

import torch
import transformers
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from spacy.lang.en.stop_words import STOP_WORDS
import codecs
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import Word2Vec
from tqdm import notebook

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import plot_confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedShuffleSplit
from sklearn.neighbors import DistanceMetric
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder, Binarizer, OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool, cv
import lightgbm as lgb  

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
torch.cuda.is_available()
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
samp_sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
RND_ST = 2202
train.info()
train.head()
train['keyword'].unique()[:5]
train['keyword'].value_counts()
len(train['keyword'].unique())
train.groupby('keyword')['keyword'].count().head(20)
def space_code_removing(df):
    
    for i in range(df.shape[0]):
        df.loc[i, 'keyword'] = re.sub(r'%20', ' ', str(df.loc[i, 'keyword']))
space_code_removing(train)
space_code_removing(test)
train['keyword'].unique()[:5]
def lemmatize(df):
    
    lemmatizer = spacy.load('en_core_web_sm')
    
    for i in range(df.shape[0]):
        lemma = lemmatizer(str(df.loc[i, 'keyword']))
        df.loc[i, 'keyword_lemma'] = " ".join([token.lemma_ for token in lemma])
topics = (train.groupby('keyword')['target']
        .agg(['count','sum'])
        .reset_index()
        .sort_values(by='count', ascending=False))
        
topics['fake'] = topics['count'] - topics['sum']
        
topics.rename(columns={'count':'total', 'sum':'true'}, inplace=True)

topics
lemmatize(topics)

len(topics['keyword_lemma'].unique())
topics.head()
topics = topics.drop('keyword', axis=1)

topics = topics.groupby('keyword_lemma')[['total','true','fake']].sum().reset_index()

topics['true_prcntg'] = (topics['true'] * 100 / topics['total']).round(2)
topics['fake_prcntg'] = (100 - topics['true_prcntg'])

topics
real_topics = topics[['keyword_lemma','true_prcntg','total']].sort_values(by='true_prcntg', ascending=False).head(10)
real_topics
fake_topics = topics[['keyword_lemma','fake_prcntg','total']].sort_values(by='fake_prcntg', ascending=False).head(10)
fake_topics
sns.set_style('whitegrid')

fig, axes = plt.subplots(1, 2, figsize=(17,5))

sns.barplot(x='true_prcntg', y='keyword_lemma', data=real_topics, color='royalblue', ax=axes[0])
sns.barplot(x='fake_prcntg', y='keyword_lemma', data=fake_topics, color='salmon', ax=axes[1])

axes[1].set_ylabel('')
axes[0].set_xlabel('Real news percentage')
axes[1].set_xlabel('Fake news percentage')

plt.suptitle('Top 10 Real and Fake news topics in train dataset', size=18);
cntrv_topics = topics[['keyword_lemma','true_prcntg']].query('48 <= true_prcntg <= 52').sort_values(by='true_prcntg', ascending=False)
cntrv_topics
temp = pd.DataFrame(train['target'].value_counts())
name = pd.Series(['real', 'fake'], name='name')
temp = temp.join(name)
temp

fig, ax = plt.subplots(figsize=(6,6))
ax.vlines(x=temp.name, ymin=0, ymax=temp['target'], color='dimgrey', alpha=0.85, linewidth=2)
ax.scatter(x=temp.name, y=temp['target'], s=75, color='firebrick', alpha=0.85)

for row in temp.itertuples():
    ax.text(row.Index, row.target+100, s=row.target, 
            horizontalalignment= 'center', verticalalignment='bottom', fontsize=10)

ax.set_title('Fake and real tweets in train dataset', size=15, y=(1.02))
ax.set_ylabel('Tweets')
ax.set_ylim(0, 5000)

#plt.tight_layout
plt.show()
for i in range(50):
    print(train.loc[i, 'text'])
    print()
def text_processing(df):
    
    text = df['text'].values
    
    documents = []

    lemmatizer = spacy.load('en_core_web_sm')
    
    df_new = df.copy()

    for sen in range(0, len(text)):
        # remove special symbols
        document = re.sub(r'\W', ' ', str(text[sen]))
    
        # remove individual symbols
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # remove individual symbols from the start of the tweet
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
        # replace few spaces to a single one
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # remove 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # convert all letters to a lower case
        document = document.lower()
    
        # spacy lemmarization
        lemma = lemmatizer(str(document))
        document = " ".join([token.lemma_ for token in lemma])
        
        # remove spacy pronouns lemmas
        #document = re.sub(r'-PRON-', '', document)
        
        df_new.loc[sen, 'text_lemm'] = document
        
    return df_new
def text_processing_02(df):
    
    text = df['text'].values
    
    df_new = df.copy()

    for sen in range(0, len(text)):
      
        ## removing part
        
        # remove hyperlinks
        document = re.sub(r'http\S+', '', str(text[sen]))
        # remove hashtags
        document = re.sub(r'#\S+', ' ', document)
        # remove special symbols
        document = re.sub(r'\W', ' ', document)
        # remove individual symbols
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # remove individual symbols from the start of the tweet
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # replace few spaces to a single one
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # remove 'b'
        document = re.sub(r'^b\s+', '', document)
        #remove ûó
        document = re.sub(r'ûó', '', document)
        
        # convert all letters to a lower case
        document = document.lower()
        
        ## replacing part
        document = re.sub(r'hwy', 'highway', document)
        document = re.sub(r'nsfw', 'not safe for work', document)
        
        
        df_new.loc[sen, 'text_lemm'] = document
        
    return df_new
temp = train.iloc[46:49].reset_index()

temp_prep = text_processing_02(temp)

temp_prep
%%time
train_lemm = text_processing_02(train)
for i in range(50):
    print(train_lemm.loc[i, 'text_lemm'])
    print()
%%time
test_lemm = text_processing_02(test)
train_lemm = train_lemm.drop(['id','keyword','location','text'], axis=1)
test_lemm = test_lemm.drop(['id','keyword','location','text'], axis=1)
X_train = train_lemm['text_lemm']
y_train = train_lemm['target']

X_test = test_lemm['text_lemm']
X_train.head()
#tfidfconverter = TfidfVectorizer(max_features=1000, 
                                 #min_df=3, max_df=0.5, 
                                 #ngram_range=(1,1),
                                 #stop_words=STOP_WORDS)

#tfidfconverter.fit(X_train)
#X_train_tf = tfidfconverter.transform(X_train)
#X_test_tf = tfidfconverter.transform(X_test)
def grid_search(model, params, features, target):
    
    search = GridSearchCV(model, params, verbose=1, cv=3, scoring='f1', n_jobs=-1)
    search.fit(features, target)
    
    print(search.best_score_)
    print(search.best_params_)  
def cross_val(model, feat, target):
    
    cvs = cross_val_score(model, feat, target, cv=5, scoring='f1').mean()
    
    return(cvs)
sgd = SGDClassifier(random_state=RND_ST)

sgd_params = dict(alpha=[1e-03, 1e-04, 1e-05, 1e-06],
                  penalty=['l1','l2'], 
                  tol=[1e-03, 1e-04, 1e-05])
#grid_search(sgd, sgd_params, X_train_tf, y_train)
#sgd = SGDClassifier(alpha=0.0001, penalty='l2', tol=0.001,  random_state=RND_ST)#
#cross_val(sgd, X_train_tf, y_train)
svm = LinearSVC(random_state = RND_ST)

svm_params = dict(C=[0.01,0.1,1,10,100],
                  max_iter=[100,250,500,1000,2000])
#grid_search(svm, svm_params, X_train_tf, y_train)
#cross_val(svm, X_train_tf, y_train)
svm = LinearSVC(C=1, max_iter=1000, random_state=RND_ST)

train_bert = train[['target','text']]
test_bert = test[['text']]

y_train = train['target']
### Downloading model and tokenizer

model_class, tokenizer_class, pretrained_weights = (
    transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
#tokenized_train = train_bert['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#tokenized_test = test_bert['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

tokenized_train = X_train.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized_test = X_test.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
def bert_features(tokenized):

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    
    attention_mask = np.where(padded != 0, 1, 0)
    
    batch_size = 1
    embeddings = []
    for i in notebook.tqdm(range(padded.shape[0] // batch_size)):
            batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)]) 
            attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])
        
            with torch.no_grad():
                batch_embeddings = model(batch, attention_mask=attention_mask_batch)
        
            embeddings.append(batch_embeddings[0][:,0,:].numpy())
    
    features = np.concatenate(embeddings)
    
    return(features)
X_train_bert = bert_features(tokenized_train) 
X_test_bert = bert_features(tokenized_test)
del tokenized_train, tokenized_test
#from numpy import savetxt 
#savetxt('/kaggle/working/X_train_bert.csv', X_train_bert, delimiter=',')
#savetxt('/kaggle/working/X_test_bert.csv', X_test_bert, delimiter=',')
X_train_bert = np.loadtxt('/kaggle/input/distilbert-preprocessed/X_train_bert.csv', delimiter=',')
X_test_bert = np.loadtxt('/kaggle/input/distilbert-preprocessed/X_test_bert.csv', delimiter=',')

y_train = train['target']
lr = LogisticRegression(random_state=RND_ST)

lr_params = {'C': np.linspace(0.0001, 100, 20),
             'max_iter':[50,100,200,500]}
grid_search(lr, lr_params, X_train_bert, y_train)
lr_final = LogisticRegression(C=15.789557894736841, max_iter=50, random_state=RND_ST)
def cat_classifier(features, target):
    
    data = Pool(data = features, 
            label = target)
    
    scores = cv(data,
            cbc_params,
            fold_count=3, 
            plot="False")
cbc_params = dict(loss_function='Logloss',
                    iterations=300,
                    learning_rate=0.07,
                    depth=4,
                    subsample=0.7,
                    verbose=100, 
                    random_state=RND_ST)
cat_classifier(X_train_bert, y_train)
### learn: 0.1747454	test: 0.4294865	best: 0.4257702 (218)	total: 1m 39s	remaining: 0us

### learn: 0.2388240	test: 0.4301056	best: 0.4288780 (350)	total: 1m 36s	remaining: 0us

### learn: 0.2695160	test: 0.4249459	best: 0.4249238 (295)	total: 1m 26s	remaining: 0us
cbc = CatBoostClassifier(loss_function='Logloss',
                    iterations=400,
                    learning_rate=0.09,
                    depth=4,
                    subsample=0.8,
                    verbose=100, 
                    random_state=RND_ST)

#cross_val(cbc, X_train_bert, y_train)
train_lgb = lgb.Dataset(X_train_bert, label=y_train, free_raw_data=False)

lgb_param = {'num_leaves': 70, 
         'objective':'binary',
         'min_data_in_leaf':23,
         'max_depth':4,
         'learning_rate':0.1,
         'num_iterations':96,
         'max_bin':3000,
         'verbosity':0,
         #'min_split_gain':90,
         'random_state':RND_ST
        }

#NUM_ROUNDS = 500
lgb_history = lgb.cv(params=lgb_param, 
                     train_set=train_lgb, 
                     metrics='cross_entropy', 
                     early_stopping_rounds=5)

len(lgb_history['cross_entropy-mean'])
lgb_history['cross_entropy-mean'][-1]
### best 0.4277043669978033
import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import EarlyStopping
def plot_hist(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
optimizer = Adam(lr=0.0001)
optimizer = SGD(lr=0.0001)
X_train_bert.shape
try:
    del model
    print('refined')
except:
    print('next')

model = Sequential()

model.add(Dense(50, input_dim=768, activation='relu', kernel_initializer='lecun_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='lecun_uniform'))

model.add(Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform'))

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit(X_train_bert, y_train, epochs=1500, validation_split=0.1, batch_size=300, verbose=0)
plot_hist(history)




def submission(model, train, target, test):
    
    model.fit(train, target)
    
    pred = model.predict(test)
    
    submission = samp_sub.copy()
    submission['target'] = pred
    
    submission.to_csv('/kaggle/working/cbcbert_15.csv', index=False)
submission(cbc, X_train_bert, y_train, X_test_bert)
def submission_lgb(params, data, test_features):
    
    lgbm = lgb.train(params, data)
    pred = lgbm.predict(test_features).round().astype('int')
    
    submission = samp_sub.copy()
    submission['target'] = pred
    
    submission.to_csv('/kaggle/working/lgbm_02.csv', index=False)
submission_lgb(lgb_param, train_lgb, X_test_bert)
del train_lgb
def nn_pred(model, X_test):

    prediction_nn = model.predict(X_test).round().astype('int')
    submission = samp_sub.copy()
    submission['target'] = prediction_nn
    
    submission.to_csv('/kaggle/working/nn_003.csv', index=False)

nn_pred(model, X_test_bert)
