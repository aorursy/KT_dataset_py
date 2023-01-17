import numpy as np
import pandas as pd
df = pd.read_csv('../input/train.csv')
df.head()
df['target'].value_counts()
count_target_0, count_target_1 = df['target'].value_counts()

df_target_0 = df[df['target'] == 0]
df_target_1 = df[df['target'] == 1]

df_target_0_under = df_target_0.sample(count_target_1)
df_under = pd.concat([df_target_0_under, df_target_1], axis=0)

df_under['target'].value_counts().plot(kind='bar', title='Count (target)')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


X_train, X_test, y_train, y_test = train_test_split(df_under['question_text'],
                                                    df_under['target'],
                                                    test_size=0.2)
tf_vectorizer = TfidfVectorizer().fit(df_under['question_text'])
X_train = tf_vectorizer.transform(X_train)
X_test = tf_vectorizer.transform(X_test)
X_train.shape
from sklearn.naive_bayes import MultinomialNB


clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)
from sklearn.metrics import f1_score


f1_score(y_test, predicted,average=None)
df_test = pd.read_csv('../input/test.csv')
X_submission = tf_vectorizer.transform(df_test['question_text'])
predicted_test = clf.predict(X_submission)

df_test['prediction'] = predicted_test
submission = df_test.drop(columns=['question_text'])
submission.head()
submission.to_csv('submission.csv', index=False)