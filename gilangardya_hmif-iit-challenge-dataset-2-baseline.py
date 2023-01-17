import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

%matplotlib inline
df = pd.read_csv('../input/train-data-2.csv')
df.head()
df = df.drop(['id'], axis=1) # drop id karena tidak penting
# pada kasus ini hanya menggunakan kolom review_sangat_singkat, header_review tidak digunakan.
X = df['review_sangat_singkat']
y = df['rating']
plt.title('Proporsi tiap kelas')
sns.countplot(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
cvec = CountVectorizer(min_df=25, ngram_range=(1,2))
tfidf_trans = TfidfTransformer()
model = LogisticRegression()

text_clf = Pipeline([
    ('vect', cvec),
    ('tfidf', tfidf_trans),
    ('clf', model),
])

text_clf.fit(X_train, y_train)
y_train_pred = text_clf.predict(X_train)
print(classification_report(y_train, y_train_pred))
print('accuracy', accuracy_score(y_train, y_train_pred))
print('rmse', np.sqrt(mean_squared_error(y_train, y_train_pred)))
y_test_pred = text_clf.predict(X_test)
print(classification_report(y_test, y_test_pred))
print('accuracy', accuracy_score(y_test, y_test_pred))
print('rmse', np.sqrt(mean_squared_error(y_test, y_test_pred)))
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
text_clf.fit(X_full, y_full)
y_full_pred = text_clf.predict(X_full)
print(classification_report(y_full, y_full_pred))
print('accuracy', accuracy_score(y_full, y_full_pred))
print('rmse', np.sqrt(mean_squared_error(y_full, y_full_pred)))
# test_data = pd.read_csv('../input/test-data-2.csv')
# test_data.head()
# test_data_pred = text_clf.predict(test_data['review_sangat_singkat'])
# test_data_pred
# submission = pd.read_csv('../input/sample-submission-2.csv')
# submission['rating'] = test_data_pred
# submission.to_csv('submission-2.csv', index=False)
