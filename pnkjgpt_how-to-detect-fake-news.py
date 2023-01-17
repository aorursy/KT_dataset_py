import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
%matplotlib inline
train = pd.read_csv('../input/fake-news-dataset/train.csv')
test = pd.read_csv('../input/fake-news-dataset/test.csv')
train.head()
train.shape
train.isnull().sum()
train['class'].value_counts()
train[train['class'] == 'February 5, 2017']
train.iloc[504, 2] = train.iloc[504, 3]
train.iloc[504, 3] = train.iloc[504, 4]
train.iloc[504, 4] = train.iloc[504, 5]
train.iloc[504, 5] = train.iloc[504, 6]
train.iloc[504, 6] = np.nan
train.iloc[[504]]
train['subject'].value_counts().plot.pie(figsize = (7, 7));
train.describe(include = 'all').T
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(train)):
    temp = re.sub('[^a-zA-Z]', ' ', train['title'][i])
    temp = temp.lower()
    temp = temp.split()
    
    temp = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp = ' '.join(temp)
    corpus.append(temp)
corpus
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
X = cv.fit_transform(corpus).toarray()
X.shape
y = train['class']
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33)
feature_names = cv.get_feature_names()
cv.get_params()
count_df = pd.DataFrame(X_train, columns = cv.get_feature_names())
count_df.head()
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train) 
pred = classifier.predict(X_val)
score = metrics.accuracy_score(y_val, pred)
print('Accuracy : %0.3f' %score)

cm = plot_confusion_matrix(classifier, X_val, y_val, cmap = 'coolwarm')
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(n_iter_no_change=50)
pac.fit(X_train, y_train)
pred = pac.predict(X_val)
score = metrics.accuracy_score(y_val, pred)
print('Accuracy : %0.3f'%score)
cm = plot_confusion_matrix(pac, X_val, y_val, cmap = 'coolwarm')
mnb = MultinomialNB(alpha = 0)
old_score = 0
for alpha in np.arange(0, 1, 0.1):
    nmnb = MultinomialNB(alpha = alpha)
    nmnb.fit(X_train, y_train)
    pred = nmnb.predict(X_val)
    score = metrics.accuracy_score(y_val, pred)
    if score > old_score:
        mnb = nmnb
    print('Alpha : {}  ||   Score : {}'.format(alpha, score))
    
#lower the value, more likely that it is Fake
sorted(zip(mnb.coef_[0], feature_names), reverse = True)
ps = PorterStemmer()
corpus = []
for i in range(len(train)):
    temp = re.sub('[^a-zA-Z]', ' ', train['text'][i])
    temp = temp.lower()
    temp = temp.split()
    
    temp = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp = ' '.join(temp)
    corpus.append(temp)
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = tf.fit_transform(corpus).toarray()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33)
feature_names = tf.get_feature_names()
feature_names
count_df = pd.DataFrame(X_train, columns=feature_names)
count_df
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train) 
pred = classifier.predict(X_val)
score = metrics.accuracy_score(y_val, pred)
print('Accuracy : %0.3f' %score)

cm = plot_confusion_matrix(classifier, X_val, y_val, cmap = 'coolwarm')
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(n_iter_no_change=50)
pac.fit(X_train, y_train)
pred = pac.predict(X_val)
score = metrics.accuracy_score(y_val, pred)
print('Accuracy : %0.3f'%score)
cm = plot_confusion_matrix(pac, X_val, y_val, cmap = 'coolwarm')
mnb = MultinomialNB(alpha = 0)
old_score = 0
for alpha in np.arange(0, 1, 0.1):
    nmnb = MultinomialNB(alpha = alpha)
    nmnb.fit(X_train, y_train)
    pred = nmnb.predict(X_val)
    score = metrics.accuracy_score(y_val, pred)
    if score > old_score:
        mnb = nmnb
    print('Alpha : {}  ||   Score : {}'.format(alpha, score))
ps = PorterStemmer()
corpus_test = []
for i in range(len(test)):
    temp = re.sub('[^a-zA-Z]', ' ', test['text'][i])
    temp = temp.lower()
    temp = temp.split()
    
    temp = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp = ' '.join(temp)
    corpus_test.append(temp)
X_t = tf.transform(corpus_test)
from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(n_iter_no_change=50)
pac.fit(X, y)
pred = pac.predict(X_t)
index = list(range(4000))
dic = {'index' : index, 'class' : pred}
df = pd.DataFrame(dic)
df.to_csv('output.csv', index = False)