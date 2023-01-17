import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
train = pd.read_csv('products_sentiment_train.tsv', sep='\t', header=None)
test = pd.read_csv('products_sentiment_test.tsv', sep='\t')
train.head()
train_feedback = train[[0]]
train_labels = train[[1]]
test_feedback = test[['text']]
test_ids = test[['Id']]
vectorizer = TfidfVectorizer(min_df=0.02, max_df=0.9, ngram_range=(3,12), sublinear_tf=True, analyzer='char_wb')
#создадим словарь грам на основе тестовых и тренировочных данных
vectorizer.fit(np.concatenate((np.array([x[0] for x in np.array(train_feedback)]), np.array([x[0] for x in np.array(test_feedback)]))))
from sklearn.svm import LinearSVC
svc = LinearSVC(penalty='l1', C=0.55, fit_intercept=False, dual=False, tol=1e-10, max_iter=100000)
from sklearn.model_selection import cross_val_score
cross_val_score(svc, vectorizer.transform(np.array([x[0] for x in np.array(train_feedback)])), train_labels, cv=10).mean()
svc.fit(vectorizer.transform(np.array([x[0] for x in np.array(train_feedback)])), train_labels)
preds = svc.predict(vectorizer.transform(np.array([x[0] for x in np.array(test_feedback)])))
ans = pd.DataFrame({'Id' : np.arange(0, len(preds)), 'y' : np.array(preds)})
ans.to_csv('submission.csv', index=False)