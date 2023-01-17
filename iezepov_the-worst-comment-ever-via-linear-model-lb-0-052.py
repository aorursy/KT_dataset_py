import re

import numpy as np

import pandas as pd

from scipy import sparse



from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer



CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Tets data contains one NaN, so we have to reaplce it with something

test.fillna(' ', inplace=True)
def normalize(text):

    text = text.lower()

    text = text.replace('\n', ' ').replace('\t', ' ')

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip()

    return text
train['normalized'] = train.comment_text.map(normalize)

test['normalized'] = test.comment_text.map(normalize)
vect_words = TfidfVectorizer(max_features=5000)

vect_chars = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(1, 3))
# Creating features

Xtrain_words = vect_words.fit_transform(train.normalized)

Xtrain_chars = vect_chars.fit_transform(train.normalized)



# Combine two different types of features into single sparse matrix

Xtrain = sparse.hstack([Xtrain_words, Xtrain_chars])
models = {}

for toxicity in CLASSES:

    # I encourage you to change this C=5.0 and try different regularization

    lm = LogisticRegression(C=5.0, random_state=42)  

    lm.fit(Xtrain, train[toxicity])

    models[toxicity] = lm

    print("Model for %s trained" % toxicity, flush=True)
Xtest_words = vect_words.transform(test.normalized)

Xtest_chars = vect_chars.transform(test.normalized)



Xtest = sparse.hstack([Xtest_words, Xtest_chars])



predictions = pd.DataFrame(test.id)

for toxicity in CLASSES:

    predictions[toxicity] = models[toxicity].predict_proba(Xtest)[:, 1]

    

predictions.to_csv('./submission.csv', index=False)
coefficients = pd.DataFrame(index=vect_words.get_feature_names() + vect_chars.get_feature_names())

for toxicity in CLASSES:

    coefficients[toxicity] = models[toxicity].coef_[0]

    

coefficients['total'] = coefficients.sum(1)

coefficients.sort_values('total', inplace=True)
# Let's randomly permutate top-10 words to amke it more interesting

np.random.seed(1)

' '.join(np.random.permutation(coefficients.head(10).index))
np.random.seed(1)

' '.join(np.random.permutation(coefficients.tail(10).index))