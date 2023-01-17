# a folk from https://www.kaggle.com/konradb/adversarial-validation
# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack, vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
# Functions
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
# read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

id_train = train['id'].copy()
id_test = test['id'].copy()

# add empty label for None
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
# fill missing values
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
# Tf-idf

# prepare tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

# create sparse matrices
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

                      smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
# combine
ytrain = np.zeros((trn_term_doc.shape[0],1)) + 1
ytest = np.zeros((test_term_doc.shape[0],1))
ydat = np.vstack((ytrain, ytest))

xdat = vstack([trn_term_doc, test_term_doc], format='csr')
nfolds = 5
xseed = 29
cval = 4

# stratified split
skf = StratifiedKFold(n_splits= nfolds, random_state= xseed)

score_vec = np.zeros((nfolds,1))
feature_importance = []

for (f, (train_index, test_index)) in enumerate(skf.split(xdat, ydat[:,0])):
    # split 
    x0, x1 = xdat[train_index], xdat[test_index]
    y0, y1 = ydat[train_index,0], ydat[test_index,0]    
    
    training_data = lgb.Dataset(x0, label = y0)
    validation_data = lgb.Dataset(x1, label = y1, reference = training_data)
    clf = lgb.train(
            params = {
                'learning_rate': 0.2,
                'num_leaves': 63,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 1,
                'objective': 'binary',
                'metric': 'auc',
                'num_threads': 4
            },
            train_set = training_data,
            num_boost_round = 200,
            valid_sets = [validation_data],
            valid_names = ['validation'],
            early_stopping_rounds = 3,
            verbose_eval = 20
        )
    feature_importance.append(pd.DataFrame({
            'fn': vec.get_feature_names(), 
            'fi':clf.feature_importance(importance_type='gain')})
        )
df = feature_importance[0]
for fis in feature_importance[1:]:
    df['fi'] += fis['fi']
df.sort_values(by = 'fi', inplace = True, ascending=False)
print("top 100 terms cover {} percent feature importances".format(
        df.iloc[:100,:]['fi'].sum() / df['fi'].sum() * 100
    ))
for term in df.iloc[:100,:]['fn']:
    print("term {:15s} appears {:10} times in train and {:10} times in test".format(
            term, 
            train['comment_text'].map(lambda x: term in " ".join(tokenize(x))).sum(), 
            test['comment_text'].map(lambda x: term in " ".join(tokenize(x))).sum())
        )