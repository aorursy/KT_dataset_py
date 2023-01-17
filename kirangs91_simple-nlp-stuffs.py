import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

import string

import re 

import nltk

nltk.download('wordnet')

data=pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')

data.info()
data.isnull().sum()
sns.set(rc={'figure.figsize':(11,4)})

pd.isnull(data).sum().plot(kind='bar')

plt.ylabel('Number of missing values')

plt.title('Missing values per Feature')
sns.countplot(x='Recommended IND',data=data)

plt.title("Distribution of Recommended IND")
data=data.dropna(subset=["Review Text"]).reset_index(drop=True)
def clean_and_tokenize(review):

    text = review.lower()

    

    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    tokens = tokenizer.tokenize(text)

    

    stemmer = nltk.stem.WordNetLemmatizer()

    text = " ".join(stemmer.lemmatize(token) for token in tokens)

    text = re.sub("[^a-z']"," ", text)

    return text

data["Clean_Review"] = data["Review Text"].apply(clean_and_tokenize)
print(data)
data.columns
print(data["Clean_Review"])
x = data['Clean_Review']

y = data['Recommended IND']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.8,random_state=0)
cv = CountVectorizer()
x_train_dtm=cv.fit_transform(x_train)

x_test_dtm=cv.transform(x_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train_dtm,y_train)
lr_pred = lr.predict(x_test_dtm)

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,lr_pred)

cm
from sklearn.metrics import classification_report
print(classification_report(y_test,lr_pred))

from sklearn.metrics import accuracy_score

accuracy_score(y_test,lr_pred)
print(x_train_dtm)
from sklearn import metrics

probs = lr.predict_proba(x_test_dtm)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()