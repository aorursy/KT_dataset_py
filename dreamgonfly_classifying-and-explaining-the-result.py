import pandas as pd
news = pd.read_csv('../input/uci-news-aggregator.csv').sample(frac=0.1)
len(news)
news.head(3)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X = news['TITLE']

y = encoder.fit_transform(news['CATEGORY'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=3)
train_vectors = vectorizer.fit_transform(X_train)

test_vectors = vectorizer.transform(X_test)



train_vectors
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20)

rf.fit(train_vectors, y_train)
from sklearn.metrics import accuracy_score
pred = rf.predict(test_vectors)

accuracy_score(y_test, pred, )