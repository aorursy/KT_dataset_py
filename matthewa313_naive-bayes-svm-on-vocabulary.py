import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
train.head()
train['project_essay_1'][0]
train['project_essay_2'][0]
lens1 = train.project_essay_1.str.len()
lens2 = train.project_essay_2.str.len()
lens3 = train.project_title.str.len()
lens4 = train.project_resource_summary.str.len()
lens1.mean(), lens1.std(), lens1.max()
lens2.mean(), lens2.std(), lens2.max()
lens3.mean(), lens3.std(), lens3.max()
lens4.mean(), lens4.std(), lens4.max()
lens1.hist(grid=False, bins=30);
lens2.hist(grid=False, bins=30);
lens3.hist(grid=False, bins=30);
lens4.hist(grid=False, bins=30);
label_cols = ['project_is_approved']
train.describe()
len(train), len(test)
train_length = len(train)
test_length = len(test)
print("Train set has ", train_length, "pieces of data.")
print("Test set has ", test_length, "pieces of data.")
ESSAY1 = 'project_essay_1'
train[ESSAY1].fillna("unknown", inplace=True)
test[ESSAY1].fillna("unknown", inplace=True)
ESSAY2 = 'project_essay_2'
train[ESSAY2].fillna("unknown", inplace=True)
test[ESSAY2].fillna("unknown", inplace=True)
TITLE = 'project_title'
train[TITLE].fillna("unknown", inplace=True)
test[TITLE].fillna("unknown", inplace=True)
RESOURCES = 'project_resource_summary'
train[RESOURCES].fillna("unknown", inplace=True)
test[RESOURCES].fillna("unknown", inplace=True)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[ESSAY1])
test_term_doc = vec.transform(test[ESSAY1])
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
preds1 = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('Fitting for ', j, '.')
    m,r = get_mdl(train[j])
    preds1[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

print(preds1[0:20])
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds1, columns = label_cols)], axis=1)
submission.to_csv('essay1_output.csv', index=False)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[ESSAY2])
test_term_doc = vec.transform(test[ESSAY2])
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
preds2 = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('Fitting for ', j)
    m,r = get_mdl(train[j])
    preds2[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    
print(preds2[0:20])
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds2, columns = label_cols)], axis=1)
submission.to_csv('essay2_output.csv', index=False)
i = 0

essaypreds = np.zeros((len(test), len(label_cols)))

for i in range(len(preds1)):
    essaypreds[i] = math.sqrt( preds1[i] * preds2[i] ) # geometric mean
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(essaypreds, columns = label_cols)], axis=1)
submission.to_csv('essaysonlysubmission.csv', index=False)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[TITLE])
test_term_doc = vec.transform(test[TITLE])
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
preds3 = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('Fitting for ', j)
    m,r = get_mdl(train[j])
    preds3[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    
print(preds3[0:20])
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds3, columns = label_cols)], axis=1)
submission.to_csv('title_output.csv', index=False)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[RESOURCES])
test_term_doc = vec.transform(test[RESOURCES])
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
preds4 = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('Fitting for ', j)
    m,r = get_mdl(train[j])
    preds4[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    
print(preds4[0:20])
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds4, columns = label_cols)], axis=1)
submission.to_csv('resourcesummary_output.csv', index=False)
i = 0

finalpreds = np.zeros((len(test), len(label_cols)))

for i in range(len(preds3)):
    finalpreds[i] = ( essaypreds[i] + preds4[i] + preds3[i] ) / 3
    
# essays = 71.515
# resource summary =
# title =
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(finalpreds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)