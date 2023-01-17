
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape,test.shape)
print(test.head())
print(train.head())
print(train.info())
print()
print(test.info())
print(train['comment_text'].value_counts())
length = train['comment_text'].str.len()
length.describe()
train_comment_text = train['comment_text']
test_comment_text = test['comment_text']
COMMENT_TEXT = pd.concat([train_comment_text, test_comment_text])
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    stop_words = 'english',
    max_features=20000)
word_vectorizer.fit(COMMENT_TEXT)
train_word_features = word_vectorizer.transform(train_comment_text)
test_word_features = word_vectorizer.transform(test_comment_text)


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=25000)
char_vectorizer.fit(COMMENT_TEXT)
train_char_features = char_vectorizer.transform(train_comment_text)
test_char_features = char_vectorizer.transform(test_comment_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

print(train_features.shape,test_features.shape)
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
predictions = {'id': test['id']}
for class_name in class_names:
    train_target = train[class_name]
    #classifier = LogisticRegression(solver='sag')
    classifier =  ExtraTreesClassifier(n_jobs=-1, random_state=3)
    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:, 1]
submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('submission_log.csv', index=False)
