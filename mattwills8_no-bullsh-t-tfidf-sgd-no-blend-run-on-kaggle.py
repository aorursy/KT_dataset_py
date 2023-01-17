import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

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
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    analyzer='word', 
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=30000)
print('Start Fit vectorizer')

tfidf = vectorizer.fit(all_comments)

print('Fit vectorizer')
print('Start transform test comments')

test_comment_features = tfidf.transform(test_comments)

print('Transformed test comments')
print('Start transform train comments')

train_comment_features = tfidf.transform(train_comments)

print('Transformed train comments')
print(train_comment_features.shape)
print(test_comment_features.shape)
submission = pd.DataFrame.from_dict({'id': test_df['id']})
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model.stochastic_gradient import SGDClassifier

scores = []
for label in labels:
    train_features = train_comment_features
    train_target = train_df[label]
    
    classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.001, random_state=42, max_iter=200, tol=None, learning_rate='optimal')
    
    X_train, X_test, y_train, y_test = train_test_split(train_features,train_target,test_size=0.33, random_state=42)
    
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    
    scores.append(score)
    print('score for {} is {}'.format(label, score))

    classifier.fit(train_features, train_target)
    submission[label] = classifier.predict_proba(test_comment_features)[:, 1]
    
    

print('Total score is {}'.format(np.mean(scores)))
submission.to_csv('submission.csv', index=False)

submission.head()
