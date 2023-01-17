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
train.head()
train['question_text'][0]
lens = train.question_text.str.len()
lens.mean(), lens.std(), lens.max()
lens.hist();
len(train),len(test)
train['question_text'].fillna("unknown", inplace=True)
test['question_text'].fillna("unknown", inplace=True)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train['question_text'])
test_term_doc = vec.transform(test['question_text'])
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
m,r = get_mdl(train['target'])

preds = m.predict_proba(test_x.multiply(r))
preds = preds[:,1] 

thresholds = np.linspace(0, 1, 1000)
score = 0.0
test_threshold=0.5
best_threshold=np.zeros(1)
best_val = np.zeros(1)

for threshold in thresholds:
    test_threshold = threshold
    max_val = np.max(val_pred[:,0])
    val_predict = (val_pred[:,0] > test_threshold)
    score = f1_score(y_val, val_predict)
    if score > best_val:
        best_threshold = threshold
        best_val = score

print("Threshold %0.6f, F1: %0.6f" % (best_threshold,best_val))
test_threshold = best_threshold

print("Best threshold: ")
print(best_threshold)
print("Best f1:")
print(best_val)
y_te = (preds > best_threshold).astype(np.int)
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)