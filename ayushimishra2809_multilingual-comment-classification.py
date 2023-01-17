import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from scipy.sparse import hstack

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.corpus import stopwords

import re

import matplotlib.pyplot as plt

import seaborn as sns

import string

import warnings

warnings.filterwarnings('ignore')
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)

valid1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

subm = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
train2=train2[:500000] # as this a too large dataset
train = pd.concat([train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0'),valid1[['comment_text', 'toxic']]]).reset_index(drop=True)
train.head(5)
test.head(5)
train['comment_text'][0]
test['content'][0]
lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist();
lens = test.content.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist()
len(train),len(test)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6))

comment_len=train[train['toxic']==1]['comment_text'].str.len()

ax1.hist(comment_len,color='cyan',linewidth=2,edgecolor='k')

ax1.set_title('toxic comments')

comment_len=train[train['toxic']==0]['comment_text'].str.len()

ax2.hist(comment_len,color='blue',linewidth=2,edgecolor='k')

ax2.set_title('Not toxic comments')

fig.suptitle('Characters in Comments',fontsize=20)



plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6))

comment_words=train[train['toxic']==1]['comment_text'].str.split().map(lambda x: len(x))

ax1.hist(comment_words,color='magenta',linewidth=2,edgecolor='k')

ax1.set_title('toxic comments')

comment_words=train[train['toxic']==0]['comment_text'].str.split().map(lambda x: len(x))

ax2.hist(comment_words,color='red',linewidth=2,edgecolor='k')

ax2.set_title('Non toxic comments')

fig.suptitle('Words in comments',fontsize=20)

plt.show()
label_cols = ['toxic']



train.describe()
train['comment_text'].fillna("unknown", inplace=True)

test['content'].fillna("unknown", inplace=True)
def lower(words):

    return words.lower()

train['comment_text']=train['comment_text'].apply(lambda x:lower(x))

def remove_numbers(words):

    return re.sub(r'\d+','',words)

train['comment_text']=train['comment_text'].apply(lambda x: remove_numbers(x))

def remove_punctuation(words):

    table=str.maketrans('','',string.punctuation)

    return words.translate(table)

train['comment_text']=train['comment_text'].apply(lambda x: remove_punctuation(x))

train['comment_text']=train['comment_text'].apply(lambda x:word_tokenize(x))

def remove_stopwords(words):

    stop_words=set(stopwords.words('english'))

    return [word for word in words if word not in stop_words]

train['comment_text']=train['comment_text'].apply(lambda x: remove_stopwords(x))

def remove_links(words):

    

    return [re.sub(r'(https?://\S+)','',word)for word in words]

train['comment_text']=train['comment_text'].apply(lambda x:remove_links(x))

def lemmatizing(words):

    lemmatizer =WordNetLemmatizer()

    return [lemmatizer.lemmatize(word) for word in words]

train['comment_text']=train['comment_text'].apply(lambda x: lemmatizing(x))

def final_text(words):

     return ' '.join(words)

train['comment_text']=train['comment_text'].apply(lambda x:final_text(x))

   

                 
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,3),

               strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

trn_term_doc = vec.fit_transform(train['comment_text'])

test_term_doc = vec.transform(test['content'])

def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc

test_x = test_term_doc
def get_model(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=4, dual=False)

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
preds = np.zeros((len(test), len(label_cols)))



for i, j in enumerate(label_cols):

    print('fit', j)

    m,r = get_model(train[j])

    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)



submission.to_csv('submission.csv', index=False)
submission.head(n=20)