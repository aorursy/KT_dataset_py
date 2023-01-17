import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import nltk

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_comments = train_df['comment_text']
test_comments = test_df['comment_text']

all_comments = pd.concat([train_comments, test_comments])

train_df.head(3)
train_df.info()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
import string
from nltk.corpus import stopwords

def clean_text(comment):
    
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    analyzer=clean_text, 
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=8000)
print('Start Fit vectorizer')

tfidf = vectorizer.fit(all_comments)

print('Fit vectorizer')
print('Start transform test comments')

test_comment_features = vectorizer.transform(test_comments)

print('Transformed test comments')
print('Start transform train comments')

train_comment_features = vectorizer.transform(train_comments)

print('Transformed train comments')
from sklearn.svm import SVC
from sklearn.linear_model.stochastic_gradient import SGDClassifier

scores = []
submission = pd.DataFrame.from_dict({'id': test_df['id']})
for class_name in labels:
    train_target = train_df[class_name]
    classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.01, random_state=42, max_iter=20, tol=None)
    
    cv_score = np.mean(cross_val_score(classifier, train_comment_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_comment_features, train_target)
    submission[class_name] = classifier.predict_proba(test_comment_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)

