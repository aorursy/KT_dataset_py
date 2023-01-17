# This code is based on the NB-SVM strong baseline kernel : https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
# I want to use random forest. It's powerful, and easy to execute with sklearn.

# From my submissions, I can see pure random forest still can't make the score higher. A better idea might be to couple it with 
# other tools, especially neural networks.

import numpy as np 
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier # import random forest classifier from sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # import two tokenizers from sklearn
# Input data files are available in the "../input/" directory.


import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')



# Any results you write to the current directory are saved as output.
train.head() # First five lines of the data
train.info()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe() # so yeah the data is very imbalanced. Very high ratio of comments aren't labeled any of the labels. 
train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True) # filling in NaN comments with "unknown"
# will use TF-IDF tokenizer to vectorize the dataset.
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
train_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['comment_text'])
train_term_doc, test_term_doc # two sparse matrices 
# Now fit the model.
preds = np.zeros((len(test), len(label_cols))) # empty np matrix to put in predictions

for i, j in enumerate(label_cols):
    print('fit', j)
    m = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=18, random_state=21)
    m.fit(train_term_doc, train[j].values)
    preds[:,i] = m.predict_proba(test_term_doc)[:,1]
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission_me.csv', index=False)
