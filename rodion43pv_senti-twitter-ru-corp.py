import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# колонки корпуса

names = ['id', 'tdate', 'tmane', 'ttext', 'ttype', 'trep', 'tfav', 'tscount', 'tfol', 'tfrien', 'listcount']
positive = pd.read_csv('../input/positive.csv', sep=';', names=names, index_col=False)

negative = pd.read_csv('../input/negative.csv', sep=';', names=names, index_col=False)

positive['sentiment'] = 1

negative['sentiment'] = 0
pos = positive[['ttext', 'sentiment']]

neg = negative[['ttext', 'sentiment']]
# выборка данных (текст, метка класса)

df = pd.concat([pos, neg], axis=0)
df.shape
# dropping ALL duplicte values 

df.drop_duplicates(subset ="ttext", 

                     keep = False, inplace = True) 
df.shape
df.to_csv('new_data.csv')
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix



from nltk.corpus import stopwords
X = df['ttext']

y = df['sentiment']

vec = CountVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(2, 4))

X_prepro = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_prepro, y, test_size=0.3, random_state=42, shuffle=True)
model = RandomForestClassifier(n_estimators=10)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
print(f"Accuracy score on train {accuracy_score(y_train, y_train_pred)}")

print(f"Accuracy score on test {accuracy_score(y_test, y_test_pred)}")

print(f"F1 score on train {f1_score(y_train, y_train_pred)}")

print(f"F1 score on test {f1_score(y_test, y_test_pred)}")
model.predict(vec.transform(['все ненавижу']))
model.predict(vec.transform(['все обожаю все круто']))
confusion_matrix(y, model.predict(X_prepro))
X = df['ttext']

y = df['sentiment']

vec = CountVectorizer(max_features=5000, ngram_range=(2, 2))

X_prepro = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_prepro, y, test_size=0.3, random_state=42, shuffle=True)



model = RandomForestClassifier(n_estimators=10)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

print(f"Accuracy score on train {accuracy_score(y_train, y_train_pred)}")

print(f"Accuracy score on test {accuracy_score(y_test, y_test_pred)}")

print(f"F1 score on train {f1_score(y_train, y_train_pred)}")

print(f"F1 score on test {f1_score(y_test, y_test_pred)}")
confusion_matrix(y, model.predict(X_prepro))