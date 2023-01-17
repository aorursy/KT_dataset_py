%%time
#%% get libraries and data
import os
import re
import string
import numpy as np 
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

numrows = None
train = pd.read_csv('../input/train.csv', index_col=['qid'], nrows=numrows)
test = pd.read_csv('../input/test.csv', index_col=['qid'], nrows=numrows)
y = train.target.values

#%% make word vectors - todo:catch numbers and punctuation, find faster tokenizer (NTLK, Spacy?)
word_vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                    min_df=3,
                                    max_df=0.9,
                                    token_pattern=r'\w{1,}',
                                    stop_words='english',
                                    max_features=50_000,
                                    strip_accents='unicode',
                                    use_idf=True,
                                    smooth_idf=True,
                                    sublinear_tf=True)

print("tokenizing")
word_vectorizer.fit(pd.concat((train['question_text'], test['question_text'])))
X = word_vectorizer.transform(train['question_text'])
X_test = word_vectorizer.transform(test['question_text'])

#%% make character vectors - coming soon



%%time
#%% Transform with Naive Bayes - combo of Ren and Jeremy Howard
class NBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1):
        self.r = None
        self.alpha = alpha

    def fit(self, X, y):
        p = self.alpha + X[y==1].sum(0)
        q = self.alpha + X[y==0].sum(0)
        self.r = csr_matrix(np.log(
            (p / (self.alpha + (y==1).sum())) /
            (q / (self.alpha + (y==0).sum()))
        ))
        return self

    def transform(self, X, y=None):
        return X.multiply(self.r)

print("nb transforming")
nbt = NBTransformer(alpha=1)
nbt.fit(X, y)
X_nb = nbt.transform(X)
X_test_nb = nbt.transform(X_test)
np.unique(X_nb.getrow(1).toarray()) #look at some contents

%%time
#%% make splits for reuse
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=911)
splits = list(skf.split(train, y))

# Logistic Regression
train_pred = np.zeros(train.shape[0])
test_pred = np.zeros(X_test.shape[0])
for train_idx, val_idx in splits:
    X_train, y_train  = X_nb[train_idx], y[train_idx]
    X_val, y_val = X_nb[val_idx], y[val_idx]
    model = LogisticRegression(solver='saga', class_weight='balanced', 
                                    C=0.5, max_iter=250, verbose=1) #seed not set
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)
    train_pred[val_idx] = val_pred[:,1]
    test_pred += model.predict_proba(X_test_nb)[:,1] / skf.get_n_splits()
    
# Topic Modeling? - coming soon

%%time
#%% find best threshold
def thresh_search(y_true, y_proba):
    best_thresh = 0
    best_score = 0
    for thresh in np.arange(0, 1, 0.01):
        score = f1_score(y_true, y_proba > thresh)
        if score > best_score:
            best_thresh = thresh
            best_score = score
    return best_thresh, best_score

print(roc_auc_score(y, train_pred))
thresh, score = thresh_search(y, train_pred)
print(thresh, score)
# submit
sub = pd.read_csv('../input/sample_submission.csv', index_col=['qid'], nrows=numrows)
sub['prediction'] = test_pred > thresh
sub.to_csv("submission.csv")