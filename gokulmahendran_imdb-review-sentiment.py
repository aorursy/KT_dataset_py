# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report,confusion_matrix,auc,f1_score,accuracy_score
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df=pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.head()
df.shape
stopwords=nltk.corpus.stopwords.words("english")
stopwords.extend(["not"])
stemmer=nltk.stem.PorterStemmer()

def clean_doc(doc):
    doc=doc.lower()
    doc=re.sub('[^a-z\s]',"",doc)
    words=doc.split(" ")
    words_imp=[stemmer.stem(word) for word in words if word not in stopwords]
    doc_cleaned=" ".join(words_imp)
    return doc_cleaned
df["sentiment"]=df["sentiment"].map({"positive":0,"negative":1})
X_train, X_test, y_train, y_test = train_test_split(df["review"].apply(clean_doc),df["sentiment"], test_size=0.3, random_state=1)

vect=CountVectorizer(min_df=10).fit(X_train)
dtm_train=vect.transform(X_train)
dtm_test=vect.transform(X_test)

model=MultinomialNB().fit(dtm_train,y_train)

pred=model.predict(dtm_test)
print(f1_score(y_test,pred))
#print(auc(y_test,pred))
df['sentiment'].value_counts()
from textblob import TextBlob
pol = lambda x: TextBlob(x).sentiment.polarity
df["pol_score"] = df["review"].apply(pol)
df["model_sentiment"]=df["pol_score"].apply(lambda x:1 if x<0 else 0)
f1_score(df["sentiment"],df["model_sentiment"])
df["sentiment"].value_counts()
df["model_sentiment"].value_counts()
accuracy_score(df["sentiment"],df["model_sentiment"])
