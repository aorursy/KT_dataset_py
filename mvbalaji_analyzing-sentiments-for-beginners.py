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
#Importing required Libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Reading the dataset
dataset = pd.read_csv('/kaggle/input/reviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head()
dataset.isnull().sum()
dataset.info()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(dataset['Liked'])
a=set(stopwords.words('english'))
from wordcloud import WordCloud
import matplotlib.pyplot as plt
stopwords = [x for x in dataset["Review"] if x not in a]
all_words = ' '.join([t for t in stopwords])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#Finding sentiment analysis (+ve, -ve and neutral)
pos = 0
neg = 0
neu = 0
for review in dataset['Review']:
    analysis = TextBlob(review)
    if analysis.sentiment[0]>0:
       pos = pos +1
    elif analysis.sentiment[0]<0:
       neg = neg + 1
    else:
       neu = neu + 1
print("Total Positive = ", pos)
print("Total Negative = ", neg)
print("Total Neutral = ", neu)
for review in dataset['Review']:
    analysis = TextBlob(review)
    print(analysis)
#Plotting sentiments
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neu]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
# Preprocessing
nltk.download('stopwords')
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in a]
    review = ' '.join(review)
    corpus.append(review)
corpus
# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
X.shape
y.shape
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4658)
#Clasiification
# Fitting Naive Bayes
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_pred_NB = NB_classifier.predict(X_test)
cm_NB = confusion_matrix(y_test, y_pred_NB)
cm_NB
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_NB)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_NB))
# Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state =4658)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
cm_RandFor = confusion_matrix(y_test, y_pred_rf)
cm_RandFor
accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))
#Support Vector Classifier
SVC_classifier = SVC(kernel = 'rbf')
SVC_classifier.fit(X_train, y_train)
y_pred_SVC = SVC_classifier.predict(X_test)
cm_SVC = confusion_matrix(y_test, y_pred_SVC)
cm_SVC
accuracy_score(y_test, y_pred_SVC)
print(classification_report(y_test, y_pred_SVC))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
confusion_matrix(y_test, y_pred_lr)
accuracy_score(y_test, y_pred_lr)
print(classification_report(y_test, y_pred_lr))
