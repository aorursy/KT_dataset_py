import pandas as pd   # manipulacao de dados do CSV
import numpy as np    # algebra linear e calculos em geral
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission_df = pd.read_csv('../input/sample_submission.csv')
train_df.head()
test_df.head()
submission_df.head()
count_by_author = train_df.groupby('author')['id'].count()
count_by_author
count_by_author.max()/count_by_author.sum()
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer()
bow.fit(test_df.append(train_df, sort=False)['text'])
print(train_df.iloc[0].text)
print(bow.vocabulary_['process'])
print(bow.vocabulary_['afforded'])
print(bow.vocabulary_['ascertaining'])
print(bow.vocabulary_['this'])
print("Tamanho do Vocabulario aprendido:", len(bow.vocabulary_))
'This' in bow.vocabulary_
train_X_bow = bow.transform(train_df['text'])
train_X_bow
x_bow_0 = train_X_bow[0].toarray().reshape(-1)
x_bow_0
idx = np.argsort(-x_bow_0)[:20]
print(idx)
print(x_bow_0[idx])
print(np.array(bow.get_feature_names())[idx])
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf.fit(train_X_bow)
train_X_tfidf = tfidf.transform(train_X_bow)
train_X_tfidf
x_tfidf_0 = train_X_tfidf[0].toarray().reshape(-1)
x_tfidf_0
idx = np.argsort(-x_tfidf_0)[:20]
print(idx)
print(x_tfidf_0[idx])
print(np.array(bow.get_feature_names())[idx])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_df['author'])
le.classes_
train_y = le.transform(train_df['author'])
train_y
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=1.0)
nb.fit(train_X_tfidf, train_y)
y_pred = nb.predict(train_X_tfidf)
y_pred
from sklearn.metrics import accuracy_score
accuracy_score(train_y, y_pred)
submission_df.head()
y_pred_proba = nb.predict_proba(train_X_tfidf)
y_pred_proba
from sklearn.metrics import log_loss
log_loss(y_pred=y_pred_proba, y_true=train_y)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
cv = StratifiedKFold(n_splits=10, random_state=42)
y_pred = cross_val_predict(MultinomialNB(alpha=1.0),
                           train_X_tfidf,
                           train_y,
                           cv=cv)
accuracy_score(train_y, y_pred)
y_pred_proba = cross_val_predict(MultinomialNB(alpha=1.0),
                                 train_X_tfidf,
                                 train_y,
                                 cv=cv,
                                 method='predict_proba')
from sklearn.metrics import log_loss
log_loss(train_y, y_pred_proba)
test_X_bow = bow.transform(test_df['text'])
test_X_bow
test_X_tfidf = tfidf.transform(test_X_bow)
test_X_tfidf
y_pred_proba = nb.predict_proba(test_X_tfidf)
y_pred_proba
submission_df.head()
submission_df['EAP'] = y_pred_proba[:, 0]
submission_df['HPL'] = y_pred_proba[:, 1]
submission_df['MWS'] = y_pred_proba[:, 2]
submission_df.head()
test_df.iloc[0].text
submission_df.to_csv('basic-submission-multonmial-nb2.csv', index=False)
