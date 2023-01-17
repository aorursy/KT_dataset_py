# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
train.shape
train.head()
test.shape
test.head()
subm.head()
subm.shape
train['comment_text'][0]
lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()
lens
lens.hist();
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train[label_cols].max(axis=1)


1-train[label_cols].max(axis=1)
train['none'] = 1-train[label_cols].max(axis=1)
train.shape
train.head()
train.describe()
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
preds = np.zeros((len(test), len(label_cols)))
preds.shape
for i, j in enumerate(label_cols):
    print('fit', j)
    #m,r = get_mdl(train[j])
    m= nb.fit(trn_term_doc, train[j])
    #preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds[:,i] = m.predict_proba(test_term_doc)[:,1]
preds
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)
