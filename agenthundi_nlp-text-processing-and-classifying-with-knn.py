import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/amazon_alexa.tsv',encoding='utf-8',delimiter='\t')
df.head()
df.info()
from nltk.tokenize import word_tokenize

from string import punctuation

from nltk.corpus import stopwords 

from nltk.stem import SnowballStemmer
def data_clean(text):

    

    line = word_tokenize(text)

    line = [word for word in line if word not in punctuation ]

    line = [word for word in line if word not in stopwords.words('english')]

    line = ' '.join(line)

    return line
x = 'Sometimes while playing a game, you can answer.'

data_clean(x)
df['clean_reviews'] = df['verified_reviews'].apply(data_clean)
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(df['clean_reviews'])
vec.vocabulary_
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df['feedback'], test_size=0.33, random_state=101)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log_pred = log.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,log_pred))

print('\n')

print(classification_report(y_test,log_pred))
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(confusion_matrix(y_test,knn_pred))

print('\n')

print(classification_report(y_test,knn_pred))