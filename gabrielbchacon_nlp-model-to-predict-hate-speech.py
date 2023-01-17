# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud



#to data preprocessing

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



#NLP tools

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer



#train split and fit models

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#model selection

from sklearn.metrics import confusion_matrix, accuracy_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/hate-speech-and-offensive-language-dataset/labeled_data.csv')

dataset.head()
dataset.info()
dataset.describe().T
dt_trasformed = dataset[['class', 'tweet']]

y = dt_trasformed.iloc[:, :-1].values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

y = np.array(ct.fit_transform(y))
print(y)
y_df = pd.DataFrame(y)

y_hate = np.array(y_df[0])

y_offensive = np.array(y_df[1])
print(y_hate)

print(y_offensive)
corpus = []

for i in range(0, 24783):

  review = re.sub('[^a-zA-Z]', ' ', dt_trasformed['tweet'][i])

  review = review.lower()

  review = review.split()

  ps = PorterStemmer()

  all_stopwords = stopwords.words('english')

  all_stopwords.remove('not')

  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

  review = ' '.join(review)

  corpus.append(review)
cv = CountVectorizer(max_features = 2000)

X = cv.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size = 0.20, random_state = 0)
classifier_np = GaussianNB()

classifier_np.fit(X_train, y_train)
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier_dt.fit(X_train, y_train)
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier_knn.fit(X_train, y_train)
classifier_lr = LogisticRegression(random_state = 0)

classifier_lr.fit(X_train, y_train)
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier_rf.fit(X_train, y_train)
#Naive Bayes

y_pred_np = classifier_np.predict(X_test)

cm = confusion_matrix(y_test, y_pred_np)

print(cm)

#Decision Tree

y_pred_dt = classifier_dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred_dt)

print(cm)

#Linear Regression

y_pred_lr = classifier_lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred_lr)

print(cm)

#Random Florest

y_pred_rf = classifier_rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)

print(cm)
rf_score = accuracy_score(y_test, y_pred_rf)

lr_score = accuracy_score(y_test, y_pred_lr)

dt_score = accuracy_score(y_test, y_pred_dt)

np_score = accuracy_score(y_test, y_pred_np)



print('Random Forest Accuracy: ', str(rf_score))

print('Linear Regression Accuracy: ', str(lr_score))

print('Decision Tree Accuracy: ', str(dt_score))

print('Naive Bayes Accuracy: ', str(np_score))