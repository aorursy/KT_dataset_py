import pandas as pd

import numpy as np

import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.head()
from wordcloud import WordCloud, STOPWORDS

import PIL

import itertools

import matplotlib.pyplot as plt



raw_str=df[df['v1']=='spam']['v2']

raw_str=' '.join(raw_str)

wordcloud = WordCloud(max_words=800,margin=0,stopwords=STOPWORDS, background_color='white',collocations=False).generate(raw_str)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
ham_str=" ".join(df[df['v1']=='ham']['v2'])

wordcloud = WordCloud( max_words=1000,margin=0,stopwords=STOPWORDS,background_color='white').generate(ham_str)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
df = df.replace(['ham','spam'],[0, 1])
X = df.iloc[:, 1].values 

y = df.v1.values
def process(x):

    processed_msg = []

 

    for i in range(0, len(x)):

        l = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ',str(x[i]))

        l=re.sub(r'[^a-zA-Z]',' ',l)

        l=re.sub(r'\s+',' ',l)

        l=l.lower()



        processed_msg.append(l)

    return processed_msg
A=process(X)
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=0)
tfidfconverter = TfidfVectorizer(max_features=3000, min_df=4, max_df=0.9, stop_words=stopwords.words('english'))  

a = tfidfconverter.fit_transform(X_train).toarray()

Xtest = tfidfconverter.transform(X_test).toarray()
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(a, y_train)
predictions = logmodel.predict(Xtest)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y_test,predictions))  

print(classification_report(y_test,predictions))  

print(accuracy_score(y_test, predictions)) #0.9659192825112107