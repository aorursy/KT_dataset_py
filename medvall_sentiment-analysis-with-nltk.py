
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
import pandas as pd
import nltk
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



url = "../input/twit-comment/vall.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = read_csv(url)

from nltk.corpus import stopwords

from stop_words import get_stop_words

stop=stopwords.words("english")

stop.extend(get_stop_words("en"))

df['text']=df.text.apply(str)

df['text']=df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))#supprimer les stopwords

df['text'].head()

print(df['text'].head())

import numpy as np
df['text'] = df['text'].replace(r'nan+', '', regex=True).replace(r'nan+', '', regex=True)
df['text'].replace('', np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)
print(df['text'])
#Remove punctuation
import string
def remove_punctuation(text):
    no_punct="".join([c for c in text if c not in string.punctuation])
    return no_punct
df['text']=df['text'].astype(str)
df['text']=df['text'].apply(lambda x: remove_punctuation(x))
df['text']
import nltk
from nltk.tokenize import word_tokenize 
x1=df.text.str.cat(sep=' ')
tokens=word_tokenize(x1)
tokens
#rendre les mots en muniscule
tokens=[x.lower()for x in tokens]
tokens
from nltk.probability import FreqDist
fdistToken = FreqDist(tokens)
fdistToken
#stop_words
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(fdistToken)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Eliminer les stopwords
f=open("../input/stopwords/stopwords","r")
li=f.readlines()
for i in range(len(li)):
    li[i]=li[i].strip()
df['text']= df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (li)]))
x2=df.text.str.cat(sep=' ')
tokens2=word_tokenize(x2)
tokens2
fdistToken2 = FreqDist(tokens2)
fdistToken2

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(fdistToken2)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

import string
def remove_tag(text):
    no_tag=" ".join([c for c in text.split() if c.startswith('@')==False])
    return no_tag
df['text']=df['text'].apply(lambda x:remove_tag(x))
df['text']
import re
def remove_digits(input_text):
        return re.sub('\d+', '', input_text)
df['text']=df['text'].apply(lambda x:remove_digits(x))
df['text']
#Stemming
from nltk.stem.porter import *
ps = PorterStemmer()
df['text'] = df['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))
df['text'].head(10)
from textblob import TextBlob

def detect_polarity(text):
    return TextBlob(text).sentiment.polarity
df['polarity']= df.text.apply(detect_polarity)
df[['text' , 'polarity']]
import matplotlib.pyplot as plt
num_bins = 50

plt.figure(figsize=(10,6))

n, bins, patches = plt.hist(df.polarity, num_bins, facecolor='blue', alpha=0.5)
df['text']

plt.xlabel('polarity')

plt.ylabel('Count')

plt.title('Histogram of polarity')

plt.show();

df.loc[ (df.polarity < 0), 'polarity'] = -1

df.loc[ (df.polarity >0), 'polarity'] = 1

import matplotlib.pyplot as plt
num_bins = 50

plt.figure(figsize=(10,6))

n, bins, patches = plt.hist(df.polarity, num_bins, facecolor='blue', alpha=0.5)

plt.xlabel('polarity')

plt.ylabel('Count')

plt.title('Histogram of polarity')

plt.show();

from sklearn.feature_extraction.text import CountVectorizer
coutvect=CountVectorizer(max_features=10)
x=coutvect.fit_transform(df['text']).toarray()
coutvect
from sklearn.model_selection import train_test_split
y=df['polarity']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()
clf1.fit(x_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

y_pred=clf1.predict(x_test)
results = confusion_matrix(y_test, y_pred)

print ('Confusion Matrix :')

print(results)

print ('Accuracy Score :',accuracy_score(y_test, y_pred) )

print ('Report : ')

print (classification_report(y_test, y_pred))
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
tv=TfidfVectorizer(min_df=0.05,max_df=0.5,max_features=1500,stop_words = 'english')
X = tv.fit_transform(df['text'])
vocab=tv.get_feature_names()

from sklearn import preprocessing
loprog = preprocessing.LabelEncoder()
r=loprog.fit_transform(df['polarity'])

from scipy.sparse import hstack
import numpy as np
X_train_dtm = hstack((X,np.array(r)[:,None]))
r
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_train_dtm,df['polarity'],random_state=20)
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

results = confusion_matrix(Y_test, y_pred)

print ('Confusion Matrix :')

print(results)

print ('Accuracy Score :',accuracy_score(Y_test, y_pred) )

print ('Report : ')

print (classification_report(Y_test, y_pred))
