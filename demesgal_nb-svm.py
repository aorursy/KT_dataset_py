from sklearn import *

import numpy as np

import pandas as pd

import glob

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



train = pd.read_csv('../input/jigsaw-train-translated-yandex-api/train_yandex.csv')

val = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv', usecols=['comment_text', 'toxic', 'lang'])
test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')

test['comment_text'] = test['content']

test['toxic'] = 0.5
val_languages = val.lang.unique().tolist()

val_languages
non_val_languages = [l for l in test.lang.unique() if l not in val_languages]

non_val_languages
train_non_val_lang = train[train.lang.isin(non_val_languages)].sample(frac = 1).reset_index()
train_non_val_lang.head(10)
toxic_count = train.toxic.sum()

print(toxic_count)

train = pd.concat([train[train.toxic == 1], train[train.toxic == 0].sample(toxic_count + 5000)]).sample(frac = 1)
[train_v, val] = train_test_split(val, test_size = 0.05, random_state = 411)
train.shape, train_v.shape
import re, string



re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 

    return re_tok.sub(r' \1 ', s).split()
# from stop_words import get_stop_words

# stop_words = get_stop_words('spanish') + get_stop_words('turkish')+ get_stop_words('italian')

# stop_words = set(stop_words)
COMMENT = 'comment_text'

LABEL = 'toxic'
binary = False
vec_train = TfidfVectorizer(ngram_range=(1,1), tokenizer=tokenize,

               min_df=2, max_df=0.9, strip_accents='unicode', use_idf=1, binary=binary,

               smooth_idf=1, sublinear_tf=1 )

vec_train.fit(train[COMMENT])

val_on_train = vec_train.transform(val[COMMENT])

trn_on_train = vec_train.transform(train[COMMENT])
vec_val = TfidfVectorizer(ngram_range=(1,1), tokenizer=tokenize,

               min_df=2, max_df=0.9, strip_accents='unicode', use_idf=1, binary=binary,

               smooth_idf=1, sublinear_tf=1 )



vec_val.fit(pd.concat([train_v[COMMENT], val[COMMENT]]))

trn_on_val = vec_val.transform(train_v[COMMENT])

val_on_val = vec_val.transform(val[COMMENT])
val_on_train, trn_on_train, trn_on_val, val_on_val
x_on_train = trn_on_train

val_x_on_train = val_on_train



x_on_val = trn_on_val

val_x_on_val = val_on_val
y_val = val[LABEL].values
def pr_train(y_i, y):

    p = x_on_train[y==y_i].sum(0)

    return (p+3) / ((y==y_i).sum()+3)



def pr_val(y_i, y):

    p = x_on_val[y==y_i].sum(0)

    return (p+0.2) / ((y==y_i).sum()+0.2)



y_train = train[LABEL].values

y_train_v = train_v[LABEL].values



r_train = np.log(pr_train(1,y_train) / pr_train(0,y_train))

r_val = np.log(pr_val(1,y_train_v) / pr_val(0,y_train_v))
x_nb_on_train = x_on_train.multiply(r_train)



C_PARAMETERS = [1.5, 2, 4]

models_train = [LogisticRegression(C=c, dual=True, solver='liblinear') for c in C_PARAMETERS]

preds_val_on_train = []



for model in models_train:

    model.fit(x_nb_on_train, y_train)

    p = model.predict_proba(val_x_on_train.multiply(r_train))[:,1]

    print(roc_auc_score(y_val, p))

    preds_val_on_train.append(p)

    

preds_ensemble_val_on_train = 2**((np.log2(preds_val_on_train[1]) + np.log2(preds_val_on_train[2]))/2)



print(roc_auc_score(y_val, preds_ensemble_val_on_train))
x_nb_on_val = x_on_val.multiply(r_val)



models_train_v = [LogisticRegression(C=c, dual=True, solver='liblinear') for c in C_PARAMETERS]

preds_val_on_val = []



for model in models_train_v:

    model.fit(x_nb_on_val, y_train_v)

    p = model.predict_proba(val_x_on_val.multiply(r_val))[:,1]

    print(roc_auc_score(y_val, p))

    preds_val_on_val.append(p)

    

preds_ensemble_val_on_val = 2**((np.log2(preds_val_on_val[1]) + np.log2(preds_val_on_val[2]))/2)



print(roc_auc_score(y_val, preds_ensemble_val_on_val))
preds_val = [

    models_train[1].predict_proba(val_x_on_train.multiply(r_train))[:,1],

    models_train[2].predict_proba(val_x_on_train.multiply(r_train))[:,1],

    models_train_v[1].predict_proba(val_x_on_val.multiply(r_val))[:,1],

    models_train_v[2].predict_proba(val_x_on_val.multiply(r_val))[:,1],

]
preds_val_ens = 2**np.mean([np.log2(p) for p in preds_val], axis = 0)

print(roc_auc_score(y_val, preds_val_ens))
is_val_lang = test.lang.isin(['tr', 'es', 'it'])
np.sum(is_val_lang.values)
test_val_lang = test.loc[is_val_lang, COMMENT]

test_non_val_lang = test.loc[~is_val_lang, COMMENT]

test_val_lang_on_train = vec_train.transform(test_val_lang)

test_nonval_lang_on_train = vec_train.transform(test_non_val_lang)

test_val_lang_on_val = vec_val.transform(test_val_lang)
test_val_lang_on_train, test_nonval_lang_on_train, test_val_lang_on_val
preds1_val_lang_on_train = models_train[0].predict_proba(test_val_lang_on_train.multiply(r_train))[:,1]

preds2_val_lang_on_train = models_train[1].predict_proba(test_val_lang_on_train.multiply(r_train))[:,1]

preds4_val_lang_on_train = models_train[2].predict_proba(test_val_lang_on_train.multiply(r_train))[:,1]



preds1_nonval_lang_on_train = models_train[0].predict_proba(test_nonval_lang_on_train.multiply(r_train))[:,1]

preds2_nonval_lang_on_train = models_train[1].predict_proba(test_nonval_lang_on_train.multiply(r_train))[:,1]

preds4_nonval_lang_on_train = models_train[2].predict_proba(test_nonval_lang_on_train.multiply(r_train))[:,1]
preds1_val_lang_on_val = models_train_v[0].predict_proba(test_val_lang_on_val.multiply(r_val))[:,1]

preds2_val_lang_on_val = models_train_v[1].predict_proba(test_val_lang_on_val.multiply(r_val))[:,1]

preds4_val_lang_on_val = models_train_v[2].predict_proba(test_val_lang_on_val.multiply(r_val))[:,1]


preds_val = 2**((np.log2(preds2_val_lang_on_train) + np.log2(preds4_val_lang_on_train) +np.log2(preds2_val_lang_on_val) + np.log2(preds4_val_lang_on_val)) / 4)

preds_nonval = 2**((np.log2(preds2_nonval_lang_on_train) + np.log2(preds4_nonval_lang_on_train)) / 2)
test.loc[is_val_lang, 'toxic'] = preds_val

test.loc[~is_val_lang, 'toxic'] = preds_nonval
test.iloc[28].toxic, test.iloc[28].content
submission1 = test[['id', 'toxic', 'lang']]

submission1.to_csv('submission1.csv', index=False)
submission2 = pd.read_csv('../input/tpu-inference-super-fast-xlmroberta/submission.csv') # Ver 4
submission1['toxic'] = submission1['toxic'] * 0.04 + submission2['toxic'] * 0.96
submission1.loc[submission1["lang"] == "es", "toxic"] *= 1.06

submission1.loc[submission1["lang"] == "fr", "toxic"] *= 1.04

submission1.loc[submission1["lang"] == "it", "toxic"] *= 0.97

submission1.loc[submission1["lang"] == "pt", "toxic"] *= 0.96

submission1.loc[submission1["lang"] == "tr", "toxic"] *= 0.98
submission1[['id', 'toxic']].to_csv('submission.csv', index=False)