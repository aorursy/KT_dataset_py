import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import os, re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
path = Path('../input/nlp-getting-started')

os.listdir(path)
train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')
print(f'Training set size: {len(train_df)}')

print(f'Test set size: {len(test_df)}')
train_df.head()
print(f'Missing values in training set: {train_df.text.isna().sum()}')

print(f'Missing values in test set: {test_df.text.isna().sum()}')
plt.bar(['No disaster', 'Disaster'], [len(train_df[train_df.target==0]), len(train_df[train_df.target==1])], color=['darkblue', 'darkorange'])

plt.xlabel('Dependent variable', fontsize=12)

plt.ylabel('Number of tweets', fontsize=12)

plt.title('Class distribution', fontsize=16)

plt.show()
print(f'Average target in training set: {np.round(train_df.target.mean(),2)}')
diff = len(train_df[train_df.target==0])-len(train_df[train_df.target==1]) 

pct_diff = np.round(diff/len(train_df),2)

diff, pct_diff
lengths_trn = train_df.text.str.len()

lengths_tst = train_df.text.str.len()

lengths_trn0 = train_df[train_df.target==0].text.str.len()

lengths_trn1 = train_df[train_df.target==1].text.str.len()

print('Avg length, min length, max length')

print('**********************************')

print(f'For training set: {int(lengths_trn.mean())}, {lengths_trn.min()}, {lengths_trn.max()}')

print(f' - no disaster tweets: {int(lengths_trn0.mean())}, {lengths_trn0.min()}, {lengths_trn0.max()}')

print(f' - disaster tweets: {int(lengths_trn1.mean())}, {lengths_trn1.min()}, {lengths_trn1.max()}')

print(f'For test set: {int(lengths_tst.mean())}, {lengths_tst.min()}, {lengths_tst.max()}')
fig, axs = plt.subplots(2, 2, sharex='row', figsize=(10,10))



axs[0, 0].hist(lengths_trn, color='darkgrey')

axs[0, 0].set_title('Training set', fontsize=16)

axs[0, 0].set_ylabel('Number of tweets', fontsize=12)

axs[0, 1].hist(lengths_tst, color='lightgrey')

axs[0, 1].set_title('Test set', fontsize=16)

axs[1, 0].hist(lengths_trn0, color='darkblue')

axs[1, 0].set_title('Training set (no disaster)', fontsize=16)

axs[1, 0].set_ylabel('Number of tweets', fontsize=12)

axs[1, 0].set_xlabel('Character lenghts', fontsize=12)

axs[1, 1].hist(lengths_trn1, color='darkorange')

axs[1, 1].set_title('Training set (disaster)', fontsize=16)

axs[1, 1].set_xlabel('Character lenghts', fontsize=12)



plt.show()
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤€‘’])')

def re_tokenizer(s): return re_tok.sub(r' \1 ', s).lower().split()
train_df.text[6]
print(re_tokenizer(train_df.text[6]))
X_train, X_valid, y_train, y_valid = train_test_split(train_df.text, train_df.target, test_size=0.1, random_state=42)
print(f'Training set size: {len(X_train)}')

print(f'Validation set size: {len(X_valid)}')
vec = CountVectorizer(ngram_range=(1,2), tokenizer=re_tokenizer, min_df=4, max_df=0.8, strip_accents='unicode', lowercase=False)
train_term_doc = vec.fit_transform(X_train)

valid_term_doc = vec.transform(X_valid)

train_term_doc.shape, valid_term_doc.shape
vocab = vec.get_feature_names()

print(f'Vocabulary size: {len(vocab)}')
# Rename term-document matrices for convenience and convert labels from pandas series into numpy arrays

x_train = train_term_doc

y_train = y_train.values

x_valid = valid_term_doc

y_valid = y_valid.values
p1 = np.squeeze(np.asarray(x_train[y_train==1].sum(0)))

p0 = np.squeeze(np.asarray(x_train[y_train==0].sum(0)))
p1.shape, p1[:10]
pr1 = (p1+1) / ((y_train==1).sum()+1)

pr0 = (p0+1) / ((y_train==0).sum()+1)
pr1.shape, pr1[:10]
vocab[2160:2170]
pr1[2164], pr0[2164]
p1[2164]/(y_train==1).sum(), (p1[2164]+1)/((y_train==1).sum()+1)
pr1[2164] / pr0[2164]
r = np.log(pr1/pr0)

r.shape, r[:10]
b = np.log((y_train==1).mean() / (y_train==0).mean()); b
preds = (x_valid @ r + b) > 0
print(f'Validation accuracy: {(preds == y_valid).mean()}')

print(f'Validation F1 score: {f1_score(y_valid, preds)}')
vec_tfidf = TfidfVectorizer(ngram_range=(1,2), tokenizer=re_tokenizer, lowercase=False,

               min_df=4, max_df=0.8, strip_accents='unicode', sublinear_tf=True)
train_term_doc_tfidf = vec_tfidf.fit_transform(X_train)

valid_term_doc_tfidf = vec_tfidf.transform(X_valid)
def pr(y_i, y, x):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
r = np.squeeze(np.asarray(np.log(pr(1, y=y_train, x=train_term_doc_tfidf)/pr(0, y=y_train, x=train_term_doc_tfidf))))
preds = (valid_term_doc_tfidf @ r + b) > 0
print(f'Validation accuracy: {(preds == y_valid).mean()}')

print(f'Validation F1 score: {f1_score(y_valid, preds)}')
nb_train = train_term_doc_tfidf.multiply(r)

nb_valid = valid_term_doc_tfidf.multiply(r)

nb_train.shape, nb_valid.shape
# Setting up the model

model = LogisticRegression(C=4, solver='liblinear')

# Fitting the model on the training data

model.fit(nb_train, y_train)

# Getting predictions for the validation set

preds = model.predict(nb_valid)
print(f'Validation accuracy: {(preds == y_valid).mean()}')

print(f'Validation F1 score: {f1_score(y_valid, preds)}')
train_term_doc_tfidf = vec_tfidf.fit_transform(train_df.text)

test_term_doc_tfidf = vec_tfidf.transform(test_df.text)

y_train = train_df.target.values
r = np.squeeze(np.asarray(np.log(pr(1, y=y_train, x=train_term_doc_tfidf)/pr(0, y=y_train, x=train_term_doc_tfidf))))
model = LogisticRegression(C=4, solver='liblinear')

model.fit(train_term_doc_tfidf.multiply(r), y_train)

preds = model.predict(test_term_doc_tfidf.multiply(r))
submit = pd.read_csv(path/'sample_submission.csv')
submit.columns
assert all(submit.id == test_df.id)

assert len(submit) == len(test_df) == len(preds)
submit.target = preds
submit.head()
# Save submissions

submit.to_csv('submission_060320.csv', index=False)