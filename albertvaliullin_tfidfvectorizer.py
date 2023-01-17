import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor, DMatrix
df_train = pd.read_csv('../input/reviews_train.csv')
# df_test = pd.read_csv('reviews_test.csv')
df_train.head(3)
helpful = pd.DataFrame(df_train['helpful'].str.strip('[]').str.split(',').tolist(), columns=['h1', 'h2']).astype(np.int64)
helpful['h1'][0]
X_train = df_train.summary.replace({np.nan: ''}) + ' ' + df_train.reviewText.replace({np.nan: ''})
# X_test = df_test.summary.replace({np.nan: ''}) + ' ' + df_test.reviewText.replace({np.nan: ''})
y_train = df_train.overall
dd = pd.DataFrame()
dd['text'] = X_train
dd['h1'] = helpful['h1']
dd['h2'] = helpful['h2']
X_train, X_test, y_train, y_test = train_test_split(dd, y_train, test_size=0.3)
# vectorizer = HashingVectorizer(stop_words=ENGLISH_STOP_WORDS, n_features=5000)
# 
# X_train_oh = vectorizer.fit_transform(X_train)
# X_test_oh = vectorizer.transform(X_test)
count_vect = CountVectorizer(max_features=75000, ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
                            # without ENGLISH_STOP_WORDS
X_train_counts = count_vect.fit_transform(X_train['text'])
X_train_counts.shape
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
X_new_counts = count_vect.transform(X_test['text'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
from scipy.sparse import hstack

X_new_tfidf = hstack((X_new_tfidf, np.array(X_test['h1'])[:,None], np.array(X_test['h2'])[:,None]))
X_train_tfidf = hstack((X_train_tfidf, np.array(X_train['h1'])[:,None], np.array(X_train['h2'])[:,None]))

X_train_tfidf
regressor = XGBRegressor(max_depth=5, n_estimators=700, objective='multi:softmax', num_class=5, learning_rate=0.25,
                         seed=42, n_jobs=-1)
regressor.fit(X_train_tfidf, y_train - 1)
y_pred = regressor.predict(X_new_tfidf) + 1
mean_absolute_error(y_test, y_pred.round())


# pd.DataFrame(data={'ID': df_test.ID, 'overall': y_pred}).to_csv('submission.csv', index=False)