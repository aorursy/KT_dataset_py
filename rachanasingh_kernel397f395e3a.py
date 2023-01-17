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
# Importing Libraries 

import numpy as np 

import pandas as pd 



# Import dataset 

df = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter = '\t') 

df.head()
df['Liked'].value_counts()
# Check for the existence of NaN values in a cell:

df.isnull().sum()
blanks = []  # start with an empty list



for i,lb,rv in df.itertuples():  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
from sklearn.model_selection import train_test_split



X = df['Review']

y = df['Liked']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC



# Naïve Bayes:

text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB())])



# Linear SVC:

text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
text_clf_nb.fit(X_train, y_train)
# Form a prediction set

predictions = text_clf_nb.predict(X_test)
# Report the confusion matrix

from sklearn import metrics

print(metrics.confusion_matrix(y_test,predictions))
# Print a classification report

print(metrics.classification_report(y_test,predictions))
# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
#Next we'll run Linear SVC
text_clf_lsvc.fit(X_train, y_train)
# Form a prediction set

predictions = text_clf_lsvc.predict(X_test)
# Report the confusion matrix

from sklearn import metrics

print(metrics.confusion_matrix(y_test,predictions))
# Print a classification report

print(metrics.classification_report(y_test,predictions))
# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))