import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_curve, roc_auc_score
train = pd.read_csv('../input/reddit_train.csv',encoding='latin-1')
test = pd.read_csv('../input/reddit_test.csv',encoding='latin-1')
train.head()
train[train.REMOVED==1][['BODY']].iloc[1,0]
train[train.REMOVED==0][['BODY']].iloc[1,0]
lens = train.BODY.str.len()
lens.mean(), lens.std(), lens.max()
lens.hist();
train.describe()
len(train),len(test)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=10, max_df=0.5, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train['BODY'])
test_term_doc = vec.transform(test['BODY'])
trn_term_doc, test_term_doc
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
preds = np.zeros((len(test), 1))

print('fitting')
m,r = get_mdl(train['REMOVED'])
preds = m.predict_proba(test_x.multiply(r))[:,1]
submission = pd.concat([pd.DataFrame(preds), test['REMOVED']], axis=1)
submission.to_csv('submission.csv', index=False)
roc_auc_score(test['REMOVED'], preds)