import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head()
dataset.shape
dataset.info()
dataset.describe()
fig=plt.figure(figsize=(5,5))

colors=["blue",'pink']

pos=dataset[dataset['Liked']==1]

neg=dataset[dataset['Liked']==0]

ck=[pos['Liked'].count(),neg['Liked'].count()]

legpie=plt.pie(ck,labels=["Positive","Negative"],

                 autopct ='%1.1f%%', 

                 shadow = True,

                 colors = colors,

                 startangle = 45,

                 explode=(0, 0.1))
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

stop=stopwords.words('english')
from wordcloud import WordCloud

positivedata = dataset[ dataset['Liked'] == 1]

positivedata =positivedata['Review']

negdata = dataset[dataset['Liked'] == 0]

negdata= negdata['Review']



def wordcloud_draw(dataset, color = 'white'):

    words = ' '.join(dataset)

    cleaned_word = " ".join([word for word in words.split()

                              if(word!='food' and word!='place')

                            ])

    wordcloud = WordCloud(stopwords=stop,

                      background_color=color,

                      width=2500,

                      height=2000

                     ).generate(cleaned_word)

    plt.figure(1,figsize=(10, 7))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    

print("Positive words are as follows")

wordcloud_draw(positivedata,'white')

print("Negative words are as follows")

wordcloud_draw(negdata)
corpus = []

for i in range(0, 1000):

  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

  review = review.lower()

  review = review.split()

  ps = PorterStemmer()

  all_stopwords = stopwords.words('english')

  all_stopwords.remove('not')

  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

  review = ' '.join(review)

  corpus.append(review)

print(corpus)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, -1].values
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn import metrics
print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegressionCV



classifier=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)



y_pred1 = classifier.predict(X_test)
print(np.concatenate((y_pred1.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test, y_pred1)

print(cm)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, y_pred1))
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

preds=classifier.predict(X_test)
print(np.concatenate((preds.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(preds,y_test)

print(cm)
print("Randon Forest Accuracy:",metrics.accuracy_score(preds,y_test))
import xgboost as xgb

xgb=xgb.XGBClassifier()

xgb.fit(X_train,y_train)
preds2=xgb.predict(X_test)

print(np.concatenate((preds2.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(preds2,y_test)

print(cm)
print("XGBoost model Accuracy:",metrics.accuracy_score(y_test, preds2))