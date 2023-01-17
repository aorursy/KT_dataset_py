import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/imdbreview/imdb_review.csv')

data
data.profile_report()
X=data['review']

y=data['sentiment']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC



text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LinearSVC(random_state=0)),

])

text_clf.fit(X_train, y_train) 
y_test_pred=text_clf.predict(X_test)



from sklearn import metrics

print(metrics.classification_report(y_test,y_test_pred))



cm = metrics.confusion_matrix(y_test,y_test_pred)

plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(y_test,y_test_pred))

plt.title(all_sample_title, size = 15);

plt.show()

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid=SentimentIntensityAnalyzer()
y_pred=X.apply(lambda review: sid.polarity_scores(review))
y_pred = y_pred.apply(lambda d:d['compound']).apply(lambda compound: 1 if compound>0 else 0)



from sklearn import metrics



print(metrics.classification_report(y, y_pred))



cm = metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(y, y_pred))

plt.title(all_sample_title, size = 15);

plt.show()