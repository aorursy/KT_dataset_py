%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
PATH='data/aclImdb/'
names = ['neg','pos']
%ls {PATH}
%ls {PATH}train
%ls {PATH}train/pos | head
trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
val,val_y = texts_labels_from_folders(f'{PATH}test',names)
trn[0]
trn_y[0]
veczr = CountVectorizer(tokenizer=tokenize)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
trn_term_doc
trn_term_doc[0]
vocab = veczr.get_feature_names(); vocab[5000:5005]
w0 = set([o.lower() for o in trn[0].split(' ')]); w0
len(w0)
veczr.vocabulary_['absurd']
trn_term_doc[0,1297]
trn_term_doc[0,5000]
def pr(y_i):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x=trn_term_doc
y=trn_y

r = np.log(pr(1)/pr(0))
b = np.log((y==1).mean() / (y==0).mean())
pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()
x=trn_term_doc.sign()
r = np.log(pr(1)/pr(0))

pre_preds = val_term_doc.sign() @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()
m = LogisticRegression(C=1e8, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()
m = LogisticRegression(C=1e8, dual=True)
m.fit(trn_term_doc.sign(), y)
preds = m.predict(val_term_doc.sign())
(preds==val_y).mean()
m = LogisticRegression(C=0.1, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()
m = LogisticRegression(C=0.1, dual=True)
m.fit(trn_term_doc.sign(), y)
preds = m.predict(val_term_doc.sign())
(preds==val_y).mean()
veczr =  CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, max_features=800000)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
trn_term_doc.shape
vocab = veczr.get_feature_names()
vocab[200000:200005]
y=trn_y
x=trn_term_doc.sign()
val_x = val_term_doc.sign()
r = np.log(pr(1) / pr(0))
b = np.log((y==1).mean() / (y==0).mean())
m = LogisticRegression(C=0.1, dual=True)
m.fit(x, y);

preds = m.predict(val_x)
(preds.T==val_y).mean()
r.shape, r
np.exp(r)
x_nb = x.multiply(r)
m = LogisticRegression(dual=True, C=0.1)
m.fit(x_nb, y);

val_x_nb = val_x.multiply(r)
preds = m.predict(val_x_nb)
(preds.T==val_y).mean()
sl=2000
# Here is how we get a model from a bag of words
md = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)
learner = md.dotprod_nb_learner()
learner.fit(0.02, 1, wds=1e-6, cycle_len=1)
learner.fit(0.02, 2, wds=1e-6, cycle_len=1)
learner.fit(0.02, 2, wds=1e-6, cycle_len=1)
