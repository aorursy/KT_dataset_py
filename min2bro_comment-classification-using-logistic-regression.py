import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
train_comment_text = train['comment_text']
test_comment_text = test['comment_text']
comment_text = pd.concat([train_comment_text, test_comment_text])
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(comment_text)
train_word_features = word_vectorizer.transform(train_comment_text)
test_word_features = word_vectorizer.transform(test_comment_text)
submission_file = pd.DataFrame.from_dict({'id': test['id']})
train.head()
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')
    classifier.fit(train_word_features, train_target)
    submission_file[class_name] = classifier.predict_proba(test_word_features)[:, 1]
submission_file