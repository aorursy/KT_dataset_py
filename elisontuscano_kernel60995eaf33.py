# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

#hide Warnings
import warnings
warnings.filterwarnings('ignore')
dataset=pd.read_csv('../input/boardgamegeek-reviews/bgg-13m-reviews.csv')
dataset.head()
dataset=dataset.iloc[:,[2,3]]
dataset
dataset.shape
dataset.dropna(subset=['comment'],inplace=True)
dataset=dataset.sample(frac=1).reset_index(drop=True)
dataset
dataset.shape
plt.hist(dataset['rating'])
plt.show()
X=dataset['comment'].values
Y=dataset['rating'].values
for index ,rate in enumerate(Y):
    Y[index]=int(round(rate))
Y
X_test, X_Sample1, Y_test, Y_Sample1 = train_test_split(X, Y, test_size = 0.1)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_Sample1, Y_Sample1, test_size = 0.25)
X_train.shape
plt.hist(dataset['rating'])
plt.show()
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(X_train, 10)
#for word, freq in common_words:
#    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['comment' , 'count'])
df1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 10 words in review before removing stop words')
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(X_train, 10)
#for word, freq in common_words:
#    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['comment' , 'count'])
df2.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 10 words in review after removing stop words')
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(X_train, 10)
#for word, freq in common_words:
#    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['comment' , 'count'])
df4.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 10 bigrams in review after removing stop words')
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(X_train, 10)
#for word, freq in common_words:
#    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['comment' , 'count'])
df6.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 10 trigrams in review after removing stop words')
#set up tfidfvectorizor
tfidf_vectorizor=TfidfVectorizer(stop_words='english', max_df=0.7,max_features=10000)
#fit and transform train and test set
tfidf_train=tfidf_vectorizor.fit_transform(X_train)
tfidf_train.shape
tfidf_dev=tfidf_vectorizor.transform(X_dev)
tfidf_dev.shape
# Fitting Naive Bayes to the Training set
NaiveClassifier = MultinomialNB()
NaiveClassifier.fit(tfidf_train, Y_train)
# Predicting the Test set results
y_pred = np.round(NaiveClassifier.predict(tfidf_dev))
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
nbscore=accuracy_score(Y_dev ,np.round(y_pred)) *100
print('Accuracy on development data : {} %'.format(nbscore))
def range_accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i] or actual[i] == (predicted[i]+1) or actual[i] == (predicted[i]-1) :
            correct += 1
    return correct / float(len(actual)) * 100.0
range_nbscore=range_accuracy_metric(Y_dev, np.round(y_pred))
print('Range Accuracy on development data : {} %'.format(range_nbscore))
print(metrics.classification_report(Y_dev,np.round(y_pred)))
from sklearn.ensemble import RandomForestClassifier
RFClassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
RFClassifier.fit(tfidf_train, Y_train)
# Predicting the Test set results
y_pred = np.round(RFClassifier.predict(tfidf_dev))
rfscore=accuracy_score(Y_dev ,np.round(y_pred)) *100
print('Accuracy on development data : {} %'.format(rfscore))
range_rfscore=range_accuracy_metric(Y_dev, np.round(y_pred))
print('Range Accuracy on development data : {} %'.format(range_rfscore))
print(metrics.classification_report(Y_dev,np.round(y_pred)))
from sklearn.linear_model import RidgeClassifier
Ridgeclassifier=RidgeClassifier()
Ridgeclassifier.fit(tfidf_train, Y_train)
# Predicting the Test set results
y_pred = np.round(Ridgeclassifier.predict(tfidf_dev))
ridge_score=accuracy_score(Y_dev ,np.round(y_pred)) *100
print('Accuracy on development data : {} %'.format(ridge_score))
range_ridge_score=range_accuracy_metric(Y_dev, np.round(y_pred))
print('Range Accuracy on development data : {} %'.format(range_ridge_score))
print(metrics.classification_report(Y_dev,np.round(y_pred)))
from sklearn.ensemble import VotingClassifier
estimators = []
estimators.append(('naive', NaiveClassifier))
estimators.append(('random', RFClassifier))
estimators.append(('ridge', Ridgeclassifier))
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(tfidf_train, Y_train)
y_pred = np.round(ensemble.predict(tfidf_dev))
ebscore=accuracy_score(Y_dev ,np.round(y_pred)) *100
print('Accuracy on development data : {} %'.format(ebscore))
range_ebscore=range_accuracy_metric(Y_dev, np.round(y_pred))
print('Range Accuracy on development data : {} %'.format(range_ebscore))
print(metrics.classification_report(Y_dev,np.round(y_pred)))
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 10000))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'softmax'))

classifier.summary()
labels = ['Naive', 'RandomForest', 'Ridge', 'Ensemble']
acc=np.round([nbscore,rfscore,ridge_score,ebscore])
rangeacc= np.round([range_nbscore,range_rfscore,range_ridge_score,range_ebscore])

x = np.arange(len(labels))
width = 0.35 

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, acc, width, label='accuracy')
rects2 = ax.bar(x + width/2, rangeacc, width, label='Range Accuracy')

ax.set_title('Bar Graph for accuracy and range accuracy')
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
from sklearn.externals import joblib
joblib.dump(Ridgeclassifier,'model/ridge_model.sav')
joblib.dump(tfidf_vectorizor,'model/tfidf_model.sav')
joblib.dump(NaiveClassifier,'model/naive_model.sav')
joblib.dump(RFClassifier,'model/randomforest_model.sav')
joblib.dump(ensemble,'model/ensemble_model.sav')
tfidf_test=tfidf_vectorizor.transform(X_test)
y_pred = np.round(Ridgeclassifier.predict(tfidf_test))
score=accuracy_score(Y_test ,np.round(y_pred)) *100
print('Accuracy on development data : {} %'.format(score))
rangescore=range_accuracy_metric(Y_test, np.round(y_pred))
print('Range Accuracy on development data : {} %'.format(rangescore))