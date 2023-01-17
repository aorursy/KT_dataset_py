import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train.head(10)
from sklearn.feature_extraction.text import CountVectorizer
%%time

vectorizer = CountVectorizer(ngram_range=(2, 2))
X_train = vectorizer.fit_transform(train['Text'])
X_train
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver = 'saga', max_iter=200)
%%time

classifier.fit(X_train, train['Sentiment'])
test = pd.read_csv('../input/test.csv')

X_test = vectorizer.transform(test['Text'])
X_test
predictions = classifier.predict_proba(X_test)
predictions
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
submission['Sentiment'] = predictions[:, 1]
submission.head()
submission.to_csv('simple_submission_baseline.csv', index=False)

