from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import re

from nltk.corpus import stopwords

import nltk

import os

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

import zipfile
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
zf1 = zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip")

train = pd.read_csv(zf1.open("labeledTrainData.tsv"), sep="\t")

zf2 = zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip")

test = pd.read_csv(zf2.open("testData.tsv"), sep="\t")
train.head()
test.head()
train.shape, test.shape
dataset=pd.concat([train, test], axis=0, sort=False)

dataset.shape
len(dataset)
dataset["review"].iloc[1]
#nltk.download('stopwords')



stops = stopwords.words('english')

#indNot = stops.index("not")

#del(stops[indNot])


corpus = []

for i in range(0, len(dataset)):

    review = re.sub('[^a-zA-Z]', ' ', dataset['review'].iloc[i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stops)]

    review = ' '.join(review)

    corpus.append(review)
'''i = 8

review = re.sub('[^a-zA-Z]', ' ', dataset['Review'].iloc[i])

review = review.lower()

review = review.split()

ps = PorterStemmer()

review = [ps.stem(word) for word in review if not word in set(stops)]

review = ' '.join(review)

corpus.append(review)'''
#corpus
cv1 = CountVectorizer(max_features = 1500)

X1 = cv1.fit_transform(corpus).toarray()

#y = dataset.iloc[:, 1]
cv2 = TfidfVectorizer(max_features = 1500)

X2 = cv2.fit_transform(corpus).toarray()

#y = dataset.iloc[:, 1]
#X1
X_train1=X1[:25000]

y_train1=dataset["sentiment"].iloc[:25000]

X_test1=X1[25000:]
X_train2=X2[:25000]

y_train2=dataset["sentiment"].iloc[:25000]

X_test2=X2[25000:]
# Fitting Naive Bayes to the Training set



classifier1 = GaussianNB()

classifier1.fit(X_train1, y_train1)

classifier2 = GaussianNB()

classifier2.fit(X_train2, y_train2)
# Predicting the Test set results

y_pred1 = classifier1.predict(X_test1)
y_pred2 = classifier2.predict(X_test2)
mysubmission2=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv")

mysubmission2["sentiment"]=y_pred2

mysubmission2.to_csv("mysubmission2.csv", index=False)
mysubmission=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv")

mysubmission["sentiment"]=y_pred1

mysubmission.to_csv("mysubmission.csv", index=False)