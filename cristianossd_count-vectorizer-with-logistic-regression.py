import numpy as np
import pandas as pd
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.head()
df['target'].value_counts()
from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(binary=True, strip_accents='unicode',
                                  max_features=90000)
vectorizer = vectorizer.fit(df['question_text'].append(df_test['question_text']))
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(df['question_text'],
                                                   df['target'],
                                                   test_size=0.2)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
X_train.shape
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression(C=1.0, multi_class='multinomial', penalty='l2',
                        solver='saga', n_jobs=1)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)
from sklearn.metrics import f1_score


[
    f1_score(y_test, predicted),
    f1_score(y_test, predicted, average='macro'),
    f1_score(y_test, predicted, average='micro'),
    f1_score(y_test, predicted, average='weighted'),
    f1_score(y_test, predicted, average=None)
]
X_submission = vectorizer.transform(df_test['question_text'])
df_test['prediction'] = clf.predict(X_submission)
submission = df_test.drop(columns=['question_text'])
submission.head()
submission.to_csv('submission.csv', index=False)