import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import train_test_split

%matplotlib inline 

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import chi2

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.utils import np_utils

import re

from keras.preprocessing import sequence

from keras.preprocessing.text import one_hot

from keras.preprocessing.text import text_to_word_sequence

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier
names=['URL','Category']

df=pd.read_csv('../input/website-classification-using-url/URL Classification.csv',names=names, na_filter=False)

dataset = df[:]


adult = dataset[1:2000]

arts = dataset[50000:52000]

business = dataset[520000:522000]

computers = dataset[535300:537300]

games = dataset[650000:652000]

health = dataset[710000:712000]

home =  dataset[764200:766200]

kids =  dataset[793080:795080]

news =  dataset[839730:841730]

recreation =  dataset[850000:852000]

reference =  dataset[955250:957250]

science =  dataset[1013000:1015000]

shopping =  dataset[1143000:1145000]

society =  dataset[1293000:1295000]

sports =  dataset[1492000:1494000]



test_data = pd.concat([adult, arts, business, computers, games, health, home, 

              kids, news, recreation, reference,science, shopping, society, sports], axis=0)



dataset.drop(dataset.index[1:2000],inplace= True)

dataset.drop(dataset.index[50000:52000],inplace= True)

dataset.drop(dataset.index[520000:522000],inplace= True)

dataset.drop(dataset.index[535300:537300],inplace= True)

dataset.drop(dataset.index[650000:652000],inplace= True)

dataset.drop(dataset.index[710000:712000],inplace= True)

dataset.drop(dataset.index[764200:766200],inplace= True)

dataset.drop(dataset.index[793080:795080],inplace= True)

dataset.drop(dataset.index[839730:841730],inplace= True)

dataset.drop(dataset.index[850000:852000],inplace= True)

dataset.drop(dataset.index[955250:957250],inplace= True)

dataset.drop(dataset.index[1013000:1015000],inplace= True)

dataset.drop(dataset.index[1143000:1145000],inplace= True)

dataset.drop(dataset.index[1293000:1295000],inplace= True)

dataset.drop(dataset.index[1492000:1494000],inplace= True)

dataset.tail()
dataset.Category.value_counts().plot(figsize=(12,5),kind='bar',color='green');

plt.xlabel('Category')

plt.ylabel('Total Number Of Individual Category')
import seaborn as sns

ax = sns.countplot(y="Category",  data=df )

plt.title("Visualization of the dataset", y=1.01, fontsize=20)

plt.ylabel("Name of the Category", labelpad=15)

plt.xlabel("Number of Categories of URLs", labelpad=15)

df[:2]
ax = sns.countplot(y = "Category",  data = dataset )

plt.title("Visualization of the train dataset", y=1.01, fontsize=20)

plt.ylabel("Name of the Category", labelpad=15)

plt.xlabel("Number of Categories of URLs", labelpad=15)
test_data.Category.value_counts().plot(figsize=(12,5),kind='bar',color='green');

plt.xlabel('Category')

plt.ylabel('Total Number Of Individual Category')
ax = sns.countplot(y = "Category",  data = test_data , color = 'gray')

plt.title("Visualization of the test dataset", y=1.01, fontsize=20)

plt.ylabel("Name of the Category", labelpad=15)

plt.xlabel("Number of Categories of URLs", labelpad=15)
X_train=dataset['URL']

y_train=dataset['Category']

#print(X_train)

X_train.shape
X_test=test_data['URL']

y_test=test_data['Category']

#print(X_test)

X_test.shape
#from sklearn.pipeline import Pipeline

#from sklearn.linear_model import SGDClassifier

#text_clf = Pipeline([

#                   ('vect', CountVectorizer( ngram_range = (2,2))),

#                    ('tfidf', TfidfTransformer()),

#                    ('clf', SGDClassifier(loss='perceptron', penalty='l2',

#                     alpha =1e-3 , max_iter=20 ,tol=None)),

#                    ])

#gs_clf = text_clf.fit(X_train , y_train)
import re

from sklearn.pipeline import Pipeline

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    

stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(3,3))

gs_clf = Pipeline([('vect', stemmed_count_vect),

                   ('tfidf', TfidfTransformer()),

                   ('clf', SGDClassifier(loss='perceptron', penalty='l2',

                    alpha =1e-4 , max_iter=20 ,tol=None)),

   ])

gs_clf = gs_clf.fit(X_train, y_train)

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix

#grid_mean_scores = [result.mean_validation_score for result in gs_clf.grid_scores_]

#print(grid_mean_scores)

y_pred=gs_clf.predict(X_test)

precision_recall_fscore_support(y_test, y_pred, average='weighted')
y_pred=gs_clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits = 4))
import seaborn as sn

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

array = confusion_matrix(y_test, y_pred)

cm=np.array(array)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cm, index = [i for i in "0123456789ABCDE"],

                  columns = [i for i in "0123456789ABCDE"])

plt.figure(figsize = (20,15))

sn.heatmap(df_cm, annot=True)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import confusion_matrix



def plot_cm(y_true, y_pred, figsize=(20,10)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)



plot_cm(y_test, y_pred)
print('Naive Bayes Train Accuracy = ',metrics.accuracy_score(y_train,gs_clf.predict(X_train)))

print('Naive Bayes Test Accuracy = ',metrics.accuracy_score(y_test,gs_clf.predict(X_test)))
print(gs_clf.predict(['http://www.businesstoday.net/']))

print(gs_clf.predict(['http://www.gamespot.net/']))
import tensorflow as tf

import pickle



# save the model to disk

filename = 'svm(2,2).sav'

pickle.dump(gs_clf, open(filename, 'wb'))

 

# some time later...

 

# load the model from disk

loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(X_test, y_test)

result