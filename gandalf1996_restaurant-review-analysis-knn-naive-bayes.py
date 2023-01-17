# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization libraries

import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

sns.set()



# NLP libraries

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, recall_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
review_df = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter='\t')

review_df.head()
corpus = []

for i in range(0,1000):

    review = re.sub('[^a-zA-Z]', ' ', review_df['Review'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).todense()

y = review_df.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.20, random_state = 0)
classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
Confusion_Matrix = confusion_matrix(y_test, y_pred)

Accuracy_Score = accuracy_score(y_test, y_pred)
df_cm = pd.DataFrame(Confusion_Matrix, range(2),

                  range(2))

#plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size

print("Accuracy Score is :", Accuracy_Score)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
Confusion_Matrix = confusion_matrix(y_test, y_pred)

Accuracy_Score = accuracy_score(y_test, y_pred)
df_cm = pd.DataFrame(Confusion_Matrix, range(2),

                  range(2))

#plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size

print("Accuracy Score is :", Accuracy_Score)