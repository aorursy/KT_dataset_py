import numpy as np

import pandas as pd



# Display columns with no limit

pd.set_option('display.max_colwidth', -1)
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
from sklearn.model_selection import train_test_split



features = ['keyword', 'location', 'text']

target = 'target'



# Split data for training and validation

X = train[features]

y = train[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, stratify=y)
import seaborn as sns



# is the dataset balanced?

print(f'{y_train.sum()} positive samples and {y_train.shape[0] - y_train.sum()} negative samples')

_ = sns.countplot(y.values)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_validate
# Define transformer and pipeline with a basic, untunned Logistic Regression

tfidf = TfidfVectorizer()

pipeline = Pipeline([('tfidf', tfidf),

                     ('clf', LogisticRegression(random_state=42))])

pipeline.fit(X_train['text'], y_train)
# Run predictions on validation data

predictions = pipeline.predict(X_val.text)

print(f'F1 score of baseline model: {f1_score(y_val, predictions)}')
test_predictions = pipeline.predict(test['text'])
submission = test[['id']].assign(target=test_predictions)

submission.head()
submission.to_csv('submission.csv', index=False)