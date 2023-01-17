import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from nltk.tokenize import RegexpTokenizer
import nltk.stem as stm
from nltk import WordNetLemmatizer, word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
import string
from time import time

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
#import textstat

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import gc
start_time = time()
color = sns.color_palette()
tqdm.pandas()
alphabet = 'abcdefghijklmnopqrstuvwxyz'
_punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']
#embeddings_index = {}
#f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
#for line in tqdm(f):
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()

#print('Found %s word vectors.' % len(embeddings_index))
def clean(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower()
    
    ## Remove stop words
    #text = text.split()
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 3]
    
    #text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub('[^a-zA-Z]',' ', text)
    text = re.sub('  +',' ',text)
    
    #text = text.split()
    #stemmer = SnowballStemmer('english')
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    return text
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def runBNB(train_X, train_y, test_X, test_y, test_X2):
    model = BernoulliNB()
    model.fit(train_X,train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def runLogistic(train_X, train_y, test_X, test_y, test_X2):
    model = LogisticRegression()
    model.fit(train_X,train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#train_X = train_df["clean_text"].fillna("_##_").values
#test_X = test_df["clean_text"].fillna("_##_").values
#print("Number of rows in train dataset : ",train_df.shape[0])
#print("Number of rows in test dataset : ",test_df.shape[0])
train_df['clean_text'] = train_df['question_text'].apply(clean)
test_df['clean_text'] = test_df['question_text'].apply(clean)
import numpy as np
def load_glove():
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    return embeddings_index

def load_para():
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)
    return embeddings_index
    
embeddings_index_1 = load_glove()
#embeddings_index_2 = load_para()
print('embedding matrix is loaded.')
def str2embeddingGlove(text,embeddings_index):
    word = text.split()
    vec = 0
    count = 0
    for w in word:
        embedding_vector = embeddings_index.get(w)
        if embedding_vector is not None:
            vec += embedding_vector
            count += 1
    return vec/(count+1)

def str2embeddingMean(text):
    word = text.split()
    vec = 0
    count = 0
    for w in word:
        embedding_vector = embeddings_index_1.get(w)
        embedding_vector1 = embeddings_index_2.get(w)
        if embedding_vector is not None:
            vec = vec + embedding_vector+embedding_vector1
            count += 2
    return vec/(count+1)
#glove embedding+lr stacking features
import gc
train_y = train_df['target']
train = np.zeros((len(train_df),300),dtype=np.float)
test = np.zeros((len(test_df),300),dtype=np.float)
for i in range(len(train_df)):
    train[i] = str2embeddingGlove(train_df['clean_text'][i],embeddings_index_1)
for i in range(len(test_df)):
    test[i] = str2embeddingGlove(test_df['clean_text'][i],embeddings_index_1)
del embeddings_index_1
gc.collect()
print('start to generate embedding features...')
k_fold = 5
cv_scores = []
LOG_pred_full_test = 0
LOG_pred_train = np.zeros((train_df.shape[0],2))
kf = model_selection.KFold(n_splits=k_fold, shuffle=True, random_state=2017)
fold = 1
for dev_index, val_index in kf.split(train):
    print(str(fold)+"-th fold is going on...")
    fold += 1
    dev_X, val_X = train[dev_index], train[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    LOG_pred_val_y, LOG_pred_test_y, LOG_model = runLogistic(dev_X, dev_y, val_X, val_y, test)
    LOG_pred_full_test = LOG_pred_full_test + LOG_pred_test_y
    LOG_pred_train[val_index,:] = LOG_pred_val_y
    cv_scores.append(metrics.log_loss(val_y, LOG_pred_val_y[:,1]))
    del dev_X,dev_y,val_X,val_y
    gc.collect()
print("Mean cv score : ", np.mean(cv_scores))
LOG_pred_full_test = LOG_pred_full_test / k_fold
train_df['glove_lr_socre'] = LOG_pred_train[:,1]
test_df['glove_lr_socre'] = LOG_pred_full_test[:,1]
eng_stopwords = set(stopwords.words("english"))   
train_df["num_words"] = train_df["question_text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["question_text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["question_text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

train_df[","] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(",")]))
test_df[","] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(",")]))

train_df[";"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(";")]))
test_df[";"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(";")]))

train_df['\"'] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split('\"')]))
test_df['\"'] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split('\"')]))

train_df["..."] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("...")]))
test_df["..."] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("...")]))

train_df["?"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("?")]))
test_df["?"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("?")]))

train_df["!"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("!")]))
test_df["!"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("!")]))

train_df["."] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(".")]))
test_df["."] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(".")]))

train_df[":"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(":")]))
test_df[":"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split(":")]))

train_df["*"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("*")]))
test_df["*"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("*")]))

train_df["-"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("-")]))
test_df["-"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split("-")]))
#train_df['syllable_analysis'] = train_df['question_text'].apply(textstat.syllable_count)
#test_df['syllable_analysis'] = test_df['question_text'].apply(textstat.syllable_count)

#train_df['reading_ease'] = train_df['question_text'].apply(textstat.flesch_reading_ease)
#test_df['reading_ease'] = test_df['question_text'].apply(textstat.flesch_reading_ease)

#train_df['flesch_level'] = train_df['question_text'].apply(textstat.flesch_kincaid_grade)
#test_df['flesch_level'] = test_df['question_text'].apply(textstat.flesch_kincaid_grade)

#train_df['fog_scale'] = train_df['question_text'].apply(textstat.gunning_fog)
#test_df['fog_scale'] = test_df['question_text'].apply(textstat.gunning_fog)

#train_df['auto_index'] = train_df['question_text'].apply(textstat.automated_readability_index)
#test_df['auto_index'] = test_df['question_text'].apply(textstat.automated_readability_index)

#train_df['coleman_index'] = train_df['question_text'].apply(textstat.coleman_liau_index)
#test_df['coleman_index'] = test_df['question_text'].apply(textstat.coleman_liau_index)

#train_df['linsear_formula'] = train_df['question_text'].apply(textstat.linsear_write_formula)
#test_df['linsear_formula'] = test_df['question_text'].apply(textstat.linsear_write_formula)

#def consensus_all(text):
#    return textstat.text_standard(text,float_output=True)

#train_df['consensus'] = train_df['question_text'].apply(consensus_all)
#test_df['consensus'] = test_df['question_text'].apply(consensus_all)
freq_words = ['best','will','good','people','way','time','year','make','sex','indian','india','women','american',
              'muslim','men','girl','black','white','think','quora','trump','chinese','many', '']

train_df['why'] = train_df['clean_text'].apply(lambda x:0 if len(str(x).split())==0 or str(x).split()[0]!='why' else 1)
test_df['why'] = test_df['clean_text'].apply(lambda x:0 if len(str(x).split())==0 or str(x).split()[0]!='why' else 1)

for word in freq_words:
    train_df[word] = train_df["clean_text"].str.count(word)
    test_df[word] = test_df["clean_text"].str.count(word)

freq_grams = ['year old','doe mean','look like','united states','feel like','high school',
             'computer science','donald trump','white people','black people','president trump',
             'trump supporter','social medium','long doe','mechanical engineer','north korea',
             'north indian','african american','hillary clinton']
for gram in freq_grams:
    train_df[gram] = train_df["clean_text"].str.count(gram)
    test_df[gram] = test_df["clean_text"].str.count(gram)
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
full_tfidf = tfidf_vec.fit_transform(train_df['clean_text'].values.tolist() + test_df['clean_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['clean_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['clean_text'].values.tolist())
train_y = train_df['target']

k_fold = 5
cv_scores = []
MNB_pred_full_test = 0
BNB_pred_full_test = 0
LOG_pred_full_test = 0
MNB_pred_train = np.zeros((train_df.shape[0],2))
BNB_pred_train = np.zeros((train_df.shape[0],2))
LOG_pred_train = np.zeros((train_df.shape[0],2))
kf = model_selection.KFold(n_splits=k_fold, shuffle=True, random_state=2017)
fold = 1
for dev_index, val_index in kf.split(train_tfidf):
    print(str(fold)+"-th fold is going on...")
    fold += 1
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    MNB_pred_val_y, MNB_pred_test_y, MNB_model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    BNB_pred_val_y, BNB_pred_test_y, BNB_model = runBNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    LOG_pred_val_y, LOG_pred_test_y, LOG_model = runLogistic(dev_X, dev_y, val_X, val_y, test_tfidf)
    MNB_pred_full_test = MNB_pred_full_test + MNB_pred_test_y
    BNB_pred_full_test = BNB_pred_full_test + BNB_pred_test_y
    LOG_pred_full_test = LOG_pred_full_test + LOG_pred_test_y
    MNB_pred_train[val_index,:] = MNB_pred_val_y
    BNB_pred_train[val_index,:] = BNB_pred_val_y
    LOG_pred_train[val_index,:] = LOG_pred_val_y
    cv_scores.append(metrics.log_loss(val_y, MNB_pred_val_y[:,1]))
print("Mean cv score : ", np.mean(cv_scores))
MNB_pred_full_test = MNB_pred_full_test / k_fold
BNB_pred_full_test = BNB_pred_full_test / k_fold
LOG_pred_full_test = LOG_pred_full_test / k_fold
train_df['tf_nb_socre'] = MNB_pred_train[:,1]
test_df['tf_nb_socre'] = MNB_pred_full_test[:,1]
train_df['tf_bb_socre'] = BNB_pred_train[:,1]
test_df['tf_bb_socre'] = BNB_pred_full_test[:,1]
train_df['tf_lr_socre'] = LOG_pred_train[:,1]
test_df['tf_lr_socre'] = LOG_pred_full_test[:,1]
import gc
del tfidf_vec,full_tfidf,train_tfidf,test_tfidf
gc.collect()
tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train_df['clean_text'].values.tolist() + test_df['clean_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['clean_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['clean_text'].values.tolist())

train_y = train_df['target']

k_fold = 5
cv_scores = []
MNB_pred_full_test = 0
BNB_pred_full_test = 0
LOG_pred_full_test = 0
MNB_pred_train = np.zeros((train_df.shape[0],2))
BNB_pred_train = np.zeros((train_df.shape[0],2))
LOG_pred_train = np.zeros((train_df.shape[0],2))
kf = model_selection.KFold(n_splits=k_fold, shuffle=True, random_state=2017)
fold = 1
for dev_index, val_index in kf.split(train_tfidf):
    print(str(fold)+"-th fold is going on...")
    fold += 1
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    MNB_pred_val_y, MNB_pred_test_y, MNB_model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    BNB_pred_val_y, BNB_pred_test_y, BNB_model = runBNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    LOG_pred_val_y, LOG_pred_test_y, LOG_model = runLogistic(dev_X, dev_y, val_X, val_y, test_tfidf)
    MNB_pred_full_test = MNB_pred_full_test + MNB_pred_test_y
    BNB_pred_full_test = BNB_pred_full_test + BNB_pred_test_y
    LOG_pred_full_test = LOG_pred_full_test + LOG_pred_test_y
    MNB_pred_train[val_index,:] = MNB_pred_val_y
    BNB_pred_train[val_index,:] = BNB_pred_val_y
    LOG_pred_train[val_index,:] = LOG_pred_val_y
    cv_scores.append(metrics.log_loss(val_y, MNB_pred_val_y[:,1]))
print("Mean cv score : ", np.mean(cv_scores))
MNB_pred_full_test = MNB_pred_full_test / k_fold
BNB_pred_full_test = BNB_pred_full_test / k_fold
LOG_pred_full_test = LOG_pred_full_test / k_fold
train_df['cv_nb_socre'] = MNB_pred_train[:,1]
test_df['cv_nb_socre'] = MNB_pred_full_test[:,1]
train_df['cv_bb_socre'] = BNB_pred_train[:,1]
test_df['cv_bb_socre'] = BNB_pred_full_test[:,1]
train_df['cv_lr_socre'] = LOG_pred_train[:,1]
test_df['cv_lr_socre'] = LOG_pred_full_test[:,1]
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

test_df['target']=np.nan
all_text = pd.concat([train_df['clean_text'],test_df['clean_text']],axis=0)
word_vect = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)

word_vect.fit(all_text)
tf_train = word_vect.transform(train_df['clean_text'])
tf_test = word_vect.transform(test_df['clean_text'])

print('TFIDF features transformation is finished!')

cols_to_drop = ['qid','question_text','clean_text']
train_X = train_df.drop(cols_to_drop+['target'],axis=1).values
train_y = train_df['target']
test_X = test_df.drop(cols_to_drop+['target'],axis=1).values

train_X = sp.hstack((train_X,tf_train,train))
test_X = sp.hstack((test_X,tf_test,test))
import lightgbm as lgb
import numpy as np

clf = lgb.LGBMClassifier(learning_rate=0.03,objective='binary',reg_alpha=0.002,
                             subsample=0.8,colsample_bytree=0.8,n_estimators=3000,
                             early_stopping_round=100,silent=-1)
clf.fit(train_X,train_y,eval_set=[(val_X,val_y)],eval_metric='binary_logloss',verbose=100,early_stopping_rounds=100)
pred_val_y = clf.predict_proba(val_X,num_iteration=clf.best_iteration_)[:,1]
pred_test_y = clf.predict_proba(test_X,num_iteration=clf.best_iteration_)[:,1]

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

pred_test_y = (pred_test_y > best_thresh).astype(int)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
len(val_X.columns)

