# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, Normalizer
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/southparklines/All-seasons.csv')
seasons = data['Season']
episodes = data['Episode']
lines = data['Line']
data
d = []
for i in data['Line']:
    d.append(str(i)[:-1])
data['Line'] = d
X = data['Line']
y = data['Character']
count = Counter(y)
names = []
X = []
y = []
i = 0
for j in count.keys():
    if list(count.values())[i] > 1000:
        names.append(list(count.keys())[i])
    i += 1
i = 0
while i < len(data['Character']):
    if data['Character'][i] in names:
        X.append(data['Line'][i])
        y.append(data['Character'][i])
    i += 1
X[:10]
y[:10]
target = y
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
cv = CountVectorizer()
tfidf = TfidfTransformer()
normalizer = Normalizer()

Xtr_bow = cv.fit_transform(X_train)
Xte_bow = cv.transform(X_test)

X_train = tfidf.fit_transform(Xtr_bow)
X_test = tfidf.fit_transform(Xte_bow)

X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)
wordcloud = WordCloud(background_color='white').generate(' '.join(lines))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#lines per character
countY = Counter(target)
plt.bar(countY.keys(), countY.values(), color='blue')
plt.title('Distribution of character lines')
plt.ylabel('Number of lines')
plt.xlabel('Character')
plt.show()

#lines per season
countS = Counter(seasons)
del countS['Season']

vals = list(countS.values())[9:]+list(countS.values())[:9]
keys = list(countS.keys())[9:]+list(countS.keys())[:9]
countS = dict(zip(keys, vals))

plt.bar(countS.keys(), countS.values(), color='red')
plt.title('Distribution of line per season')
plt.ylabel('Number of lines')
plt.xlabel('Season')
plt.show()

#lines per episode
countE = Counter(episodes)
del countE['Episode']

plt.bar(countE.keys(), countE.values(), color='green')
plt.title('Distribution of lines per episode')
plt.ylabel('Number of lines')
plt.xlabel('Episode in season')
plt.show()
models = [MultinomialNB(), LinearSVC(), PassiveAggressiveClassifier()]

for model in models:
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    score = model.score(X_test, y_test)
    cross_val = cross_val_score(model, X_test, y_test).mean()

    print(model, 'accuracy:', accuracy, ' score:', score, ' cross_val:', cross_val)
model = LinearSVC(C=0.4, loss='squared_hinge', penalty='l2', tol=0.1, multi_class='ovr')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
score = model.score(X_test, y_test)
cross_val = cross_val_score(model, X_test, y_test).mean()

print('accuracy:', accuracy, ' score:', score, ' cross_val:', cross_val)