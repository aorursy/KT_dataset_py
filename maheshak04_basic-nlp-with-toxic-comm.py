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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample=pd.read_csv('../input/sample_submission.csv')
train.head()
test.head()
sample.head()
train['comment_text'] = train['comment_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['comment_text'].head()
test['comment_text'] = test['comment_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
test['comment_text'].head()
train['comment_text'] = train['comment_text'].str.replace('[^\w\s]','')
train['comment_text'].head()
test['comment_text'] = test['comment_text'].str.replace('[^\w\s]','')
test['comment_text'].head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['comment_text'] = train['comment_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['comment_text'].head()

test['comment_text'] = test['comment_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test['comment_text'].head()
from textblob import Word
train['comment_text'] = train['comment_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['comment_text'].head()
test['comment_text'] = test['comment_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
test['comment_text'].head()
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc
test_x = test_term_doc
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
from sklearn.linear_model import LogisticRegression
preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': sample["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)
