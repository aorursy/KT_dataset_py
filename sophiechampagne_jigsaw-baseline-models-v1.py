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
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import gc
import os

from sklearn.model_selection import train_test_split
from collections import Counter

from gensim.models.phrases import Phraser,Phrases 
from gensim.models.word2vec import Word2Vec 
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
# Loading data

train_en = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_en['lang'] = 'en'

train_es = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')
train_es['lang'] = 'es'

train_fr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')
train_fr['lang'] = 'fr'

train_pt = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')
train_pt['lang'] = 'pt'

train_ru = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')
train_ru['lang'] = 'ru'

train_it = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')
train_it['lang'] = 'it'

train_tr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')
train_tr['lang'] = 'tr'

#train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
#train2.toxic = train2.toxic.round().astype(int)
#train2['lang'] = 'en'

data = pd.concat([
    
    train_en[['comment_text', 'lang', 'toxic']],
    train_es[['comment_text', 'lang', 'toxic']],
    train_tr[['comment_text', 'lang', 'toxic']],
    train_fr[['comment_text', 'lang', 'toxic']],
    train_pt[['comment_text', 'lang', 'toxic']],
    train_ru[['comment_text', 'lang', 'toxic']],
    train_it[['comment_text', 'lang', 'toxic']]
    
]).sample(n=100000).reset_index(drop=True)

del train_en, train_es, train_fr, train_pt, train_ru, train_it, train_tr
gc.collect()
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
submission = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
validation.describe()
data.shape,validation.shape,submission.shape
data.head()
data['comment_text'][0]
data['comment_text'][100]
lens = data.comment_text.str.len()
lens.min(),lens.mean(), lens.std(), lens.max()
len(data),len(validation)
data['comment_text'].fillna("unknown", inplace=True)
validation['comment_text'].fillna("unknown", inplace=True)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
#clean txt function 
#write a class function for cleaning and preprocss the doctoVec attributes 
#clean the text using genism 
from gensim import utils
import gensim.parsing.preprocessing as gsp

#the filters are doing the following 
#remove tags 
#remove punctuation
#standarized the spaces 
#stop words 
#stemming 
#lower case for all words 
filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text(s):
    s = str(s).lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s
submission.head()
n = data.shape[0]

#clean the text first 
data["clean_comment"] = data.comment_text.apply(clean_text)
validation["clean_comment"] = validation.comment_text.apply(clean_text)
submission["clean_comment"] = submission.content.apply(clean_text)

#split the train and test from the whole table 
x_train, x_test, y_train, y_test = train_test_split(data["clean_comment"],data.toxic, test_size=0.3)

#vectorization of the model 
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
train_term_doc = vec.fit_transform(x_train)
test_term_doc = vec.transform(x_test)
valid_term_doc = vec.transform(validation['clean_comment'])
submission_term_doc = vec.transform(submission['clean_comment'])
train_term_doc.shape
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression(n_jobs=1, C=4)
logreg.fit(train_term_doc, y_train)
y_pred = logreg.predict(test_term_doc)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_test, y_pred))
#score the validation dataset
y_valid = validation['toxic']
y_pred_valid = logreg.predict(valid_term_doc)
print('Testing accuracy %s' % accuracy_score(y_valid, y_pred_valid))
print('Testing F1 score: {}'.format(f1_score(y_valid, y_pred_valid, average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_valid, y_pred_valid))
#score the submission file 
y_pred_submission = logreg.predict(submission_term_doc)
#load the sample submission file 
sample_sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
submid = pd.DataFrame({'id': sample_sub["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_submission,columns=['toxic'])], axis=1)

submission.to_csv('submission.csv', index=False)
#run the xgboost 
xgboost_model = xgb.train({'max_depth': 5, "seed": 123,'objective':'binary:logistic',
                   'learning_rate':0.23,'min_child_weight':4}, 
                  xgb.DMatrix(train_term_doc, label=y_train), num_boost_round=500)

y_pred_xgboost = xgboost_model.predict(xgb.DMatrix(test_term_doc, label=y_test))

print('Testing accuracy %s' % accuracy_score(y_test, y_pred_xgboost.round()))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred_xgboost.round(), average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_test, y_pred_xgboost))
#validation set
y_valid = validation['toxic']
y_pred_valid = xgboost_model.predict(xgb.DMatrix(valid_term_doc, label=y_valid))
#print('Testing accuracy %s' % accuracy_score(y_valid, y_pred_valid))
#print('Testing F1 score: {}'.format(f1_score(y_valid, y_pred_valid, average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_valid, y_pred_valid))
#score the submissint file 
y_pred_submission = xgboost_model.predict(xgb.DMatrix(submission_term_doc))
sample_sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
submid = pd.DataFrame({'id': sample_sub["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_submission,columns=['toxic'])], axis=1)

submission.to_csv('submission.csv', index=False)
#use the English Version
data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
validation = pd.read_csv('../input/translated/validation_translate.csv')
submission = pd.read_csv('../input/translated/test_translate.csv')
n = data.shape[0]

#clean the text first 
data["clean_comment"] = data.comment_text.apply(clean_text)
validation["clean_comment"] = validation.comment_text.apply(clean_text)
submission["clean_comment"] = submission.content.apply(clean_text)

#split the train and test from the whole table 
x_train, x_test, y_train, y_test = train_test_split(data["clean_comment"],data.toxic, test_size=0.3)

#vectorization of the model 
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
train_term_doc = vec.fit_transform(x_train)
test_term_doc = vec.transform(x_test)
valid_term_doc = vec.transform(validation['clean_comment'])
submission_term_doc = vec.transform(submission['clean_comment'])
logreg = LogisticRegression(n_jobs=1, C=4)
logreg.fit(train_term_doc, y_train)
y_pred = logreg.predict(test_term_doc)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_test, y_pred))
#score the validation dataset
y_valid = validation['toxic']
y_pred_valid = logreg.predict(valid_term_doc)
print('Testing accuracy %s' % accuracy_score(y_valid, y_pred_valid))
print('Testing F1 score: {}'.format(f1_score(y_valid, y_pred_valid, average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_valid, y_pred_valid))
sample_sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
submid = pd.DataFrame({'id': sample_sub["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_submission,columns=['toxic'])], axis=1)

submission.to_csv('submission.csv', index=False)
x_train.shape
#train the xgboost model 
#run the xgboost 
xgboost_model = xgb.train({'max_depth': 5, "seed": 123,'objective':'binary:logistic',
                   'learning_rate':0.23,'min_child_weight':4}, 
                  xgb.DMatrix(train_term_doc, label=y_train), num_boost_round=500)

y_pred_xgboost = xgboost_model.predict(xgb.DMatrix(test_term_doc, label=y_test))

print('Testing accuracy %s' % accuracy_score(y_test, y_pred_xgboost.round()))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred_xgboost.round(), average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_test, y_pred_xgboost))
#validation set
y_valid = validation['toxic']
y_pred_valid = xgboost_model.predict(xgb.DMatrix(valid_term_doc, label=y_valid))
#print('Testing accuracy %s' % accuracy_score(y_valid, y_pred_valid))
#print('Testing F1 score: {}'.format(f1_score(y_valid, y_pred_valid, average='weighted')))
print('Testing AUC score %s' % roc_auc_score(y_valid, y_pred_valid))
#score the submissint file 
y_pred_submission = xgboost_model.predict(xgb.DMatrix(submission_term_doc))
sample_sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
submid = pd.DataFrame({'id': sample_sub["id"]})
submission = pd.concat([submid, pd.DataFrame(y_pred_submission,columns=['toxic'])], axis=1)

submission.to_csv('submission.csv', index=False)
#Navie-Bayes-SVM version- use multi-language
# Loading data

train_en = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_en['lang'] = 'en'

train_es = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')
train_es['lang'] = 'es'

train_fr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')
train_fr['lang'] = 'fr'

train_pt = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')
train_pt['lang'] = 'pt'

train_ru = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')
train_ru['lang'] = 'ru'

train_it = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')
train_it['lang'] = 'it'

train_tr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')
train_tr['lang'] = 'tr'

#train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
#train2.toxic = train2.toxic.round().astype(int)
#train2['lang'] = 'en'

train = pd.concat([
    
    train_en[['comment_text', 'lang', 'toxic']],
    train_es[['comment_text', 'lang', 'toxic']],
    train_tr[['comment_text', 'lang', 'toxic']],
    train_fr[['comment_text', 'lang', 'toxic']],
    train_pt[['comment_text', 'lang', 'toxic']],
    train_ru[['comment_text', 'lang', 'toxic']],
    train_it[['comment_text', 'lang', 'toxic']]
    
]).sample(n=500000).reset_index(drop=True)

del train_en, train_es, train_fr, train_pt, train_ru, train_it, train_tr
gc.collect()
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub= pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
label_cols = ['toxic']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )


trn_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['content'])
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': sub["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)