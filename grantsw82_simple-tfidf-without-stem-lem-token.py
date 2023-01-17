import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')

print('Imports Complete')
#Here is where I read my data into pandas dataframes
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


train_df.head()

from matplotlib import pyplot
import numpy as np
%matplotlib inline

count_target_0, count_target_1 = train_df['target'].value_counts()

train_df_target_0 = train_df[train_df['target'] == 0]
train_df_target_1 = train_df[train_df['target'] == 1]

train_df_target_0_under = train_df_target_0.sample(count_target_1)
train_df_under = pd.concat([train_df_target_0_under, train_df_target_1], axis=0)

train_df_under['target'].value_counts().plot(kind='bar', title='Count (target)')
train_df_under['target'].value_counts()
sam_train_under = train_df_under.sample(50000)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sam_train_under[['question_text']], sam_train_under['target'], test_size=0.2)
tfidf_vect = TfidfVectorizer(stop_words='english')
tfidf_vect_fit = tfidf_vect.fit(X_train['question_text'])

tfidf_train = tfidf_vect_fit.transform(X_train['question_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['question_text'])

X_train_vect = pd.DataFrame(tfidf_train.toarray())
X_test_vect =  pd.DataFrame(tfidf_test.toarray())

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time
sgd = SGDClassifier()

start = time.time()
sgd_model = sgd.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = sgd_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1, average='binary')
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
### Try Logistic regression
lr = LogisticRegression(C=0.1, solver='sag')

start = time.time()
lr_model = lr.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = lr_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1, average='binary')
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))

X_submission = tfidf_vect.transform(test_df['question_text'])
predicted_test = sgd.predict(X_submission)

test_df['prediction'] = predicted_test
submission = test_df.drop(columns=['question_text'])
submission.head()
submission.to_csv('submission.csv', index=False)
