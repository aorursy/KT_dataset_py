# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()



from sklearn.feature_extraction.text import CountVectorizer



import matplotlib.pyplot as plt
df =pd.read_csv('../input/fake-news/train.csv')
df.head()
x= df.drop('label',axis=1)
x.head(2)
y = df['label']
df.shape
df.info()
df.isnull().sum()
df=df.dropna()
df.head()
df['title'][3]
messeges =df.copy()
messeges.reset_index(inplace=True)
messeges.head()
corpus = []

for i in range(0, len(messeges)):

    review = re.sub('[^a-zA-Z]', ' ', messeges['title'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
corpus[6]
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))

X = cv.fit_transform(corpus).toarray()
# show resulting vocabulary; the numbers are not counts, they are the position in the sparse vector.

cv.vocabulary_
X.shape
y=messeges['label']
## Divide the dataset into Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
cv.get_feature_names()[:20]
cv.get_params()
count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
count_df.head()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    See full source and example: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.naive_bayes import MultinomialNB

classifier=MultinomialNB()

from sklearn import metrics

import numpy as np

import itertools
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred)

print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, pred)

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred)

score
y_train.shape
classifier=MultinomialNB(alpha=0.1)
previous_score=0

for alpha in np.arange(0,1,0.1):

    sub_classifier=MultinomialNB(alpha=alpha)

    sub_classifier.fit(X_train,y_train)

    y_pred=sub_classifier.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    if score>previous_score:

        classifier=sub_classifier

    print("Alpha: {}, Score : {}".format(alpha,score))
## Get Features names

feature_names = cv.get_feature_names()
classifier.coef_[0]
### Most real

sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]
### Most fake

sorted(zip(classifier.coef_[0], feature_names))[:5000]
train=pd.read_csv('../input/fake-news/train.csv')

test=pd.read_csv('../input/fake-news/test.csv')

test.info()

test['label']='t'

train.info()
from sklearn.feature_extraction.text import TfidfTransformer



test=test.fillna(' ')

train=train.fillna(' ')

test['total']=test['title']+' '+test['author']+test['text']

train['total']=train['title']+' '+train['author']+train['text']



#tfidf

transformer = TfidfTransformer(smooth_idf=False)

count_vectorizer = CountVectorizer(ngram_range=(1, 2))

counts = count_vectorizer.fit_transform(train['total'].values)

tfidf = transformer.fit_transform(counts)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer



#data prep

test=test.fillna(' ')

train=train.fillna(' ')

test['total']=test['title']+' '+test['author']+test['text']

train['total']=train['title']+' '+train['author']+train['text']



#tfidf

transformer = TfidfTransformer(smooth_idf=False)

count_vectorizer = CountVectorizer(ngram_range=(1, 2))

counts = count_vectorizer.fit_transform(train['total'].values)

tfidf = transformer.fit_transform(counts)
targets = train['label'].values

test_counts = count_vectorizer.transform(test['total'].values)

test_tfidf = transformer.fit_transform(test_counts)



#split in samples

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,

                              AdaBoostClassifier)



Extr = ExtraTreesClassifier(n_estimators=5,n_jobs=4)

Extr.fit(X_train, y_train)

print('Accuracy of ExtrTrees classifier on training set: {:.2f}'

     .format(Extr.score(X_train, y_train)))

print('Accuracy of Extratrees classifier on test set: {:.2f}'

     .format(Extr.score(X_test, y_test)))
from sklearn.naive_bayes import MultinomialNB



NB = MultinomialNB()

NB.fit(X_train, y_train)

print('Accuracy of NB  classifier on training set: {:.2f}'

     .format(NB.score(X_train, y_train)))

print('Accuracy of NB classifier on test set: {:.2f}'

     .format(NB.score(X_test, y_test)))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)

print('Accuracy of Lasso classifier on training set: {:.2f}'

     .format(logreg.score(X_train, y_train)))

print('Accuracy of Lasso classifier on test set: {:.2f}'

     .format(logreg.score(X_test, y_test)))
targets = train['label'].values

logreg = LogisticRegression()

logreg.fit(counts, targets)



example_counts = count_vectorizer.transform(test['total'].values)

predictions = logreg.predict(example_counts)

pred=pd.DataFrame(predictions,columns=['label'])

pred['id']=test['id']

pred.groupby('label').count()
pred.to_csv('countvect5.csv', index=False)