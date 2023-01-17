import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import StandardScaler

import os
os.environ['OMP_NUM_THREADS'] = '4'
print(os.listdir("../input"))
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,2),
    max_features=20000)
word_vectorizer.fit(all_text)
print("fitted...")
train_word_features = word_vectorizer.transform(train_text)
print("train...")
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2,6),
    max_features=50000)
char_vectorizer.fit(all_text)
print("fitted...")
train_char_features = char_vectorizer.transform(train_text)
print("train...")
test_char_features = char_vectorizer.transform(test_text)
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    print(class_name)
    train_target = train[class_name]
    classifier = LogisticRegression()
    
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

submission.to_csv('submission.csv', index=False)
