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
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
false = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true.head(10)
false.head()
# lets add Target column into both datasets

true['target'] = 1
false['target'] = 0
# lets combine our Dataset

data = pd.concat([true, false], axis= 0)
data.head()
data.info()
import seaborn as sns

sns.countplot(data['target'])
a = data['target'].value_counts()
import plotly.express as px
fig = px.bar(a)
fig.show()
# a =  data['title'].astype('str')
# a
data['combine'] = data['title'] + data['text'] + data['subject']
data['combine'] = data['combine'].astype('str')
data
# let us divide our data into given data and target column

train = data['combine'].astype('str')
train
target = data.target
target
# text cleaning


import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

lement = WordNetLemmatizer()
stemr = PorterStemmer()
stopword = stopwords.words('english')
def text_preprocessing(texts):
    tex=texts.strip()
    
    texts_word=[word for word in tex.split() if "@" not in word]
    tex=" ".join(texts_word)
#     texts_word=[word for word in tex.split() if "@" not in word]
#     tex=" ".join(texts_word)
    
    texts_word=[word for word in tex.split() if "$" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "^" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "_" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "=" not in word]
    tex=" ".join(texts_word)
    
    
    texts_word=[word for word in tex.split() if "www." not in word]
    tex=" ".join(texts_word)
    
    texts_word=[word for word in tex.split() if "http" not in word]
    tex=" ".join(texts_word)
    
    
    texts_word=[word for word in tex if word not in string.punctuation]
    tex="".join(texts_word)
    
    
    texts_word=[word for word in tex.split() if word not in stopword]
    tex=" ".join(texts_word)
    
    texts_word = [lement.lemmatize(word) for word in tex.split()]
    tex = " ".join(texts_word)
    
    texts_word = [stemr.stem(word) for word in tex.split()]
    tex = " ".join(texts_word)
        
    
    texts_word=[word for word in tex.split() if word.isalpha()]
    tex=" ".join(texts_word)
    
    
    texts_word=[word.lower() for word in tex.split()]
    tex=" ".join(texts_word)
    tex=tex.strip()
    return tex.split()

a = 'this si not going !@#$ hello okay sports usa'
cv=CountVectorizer(analyzer=text_preprocessing).fit(train)

cv_trans=cv.transform(train)
cv_trans
tfidf=TfidfTransformer().fit(cv_trans)
tfidf_trans=tfidf.transform(cv_trans)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_trans , target, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

np.mean(predicted == y_test)
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier().fit(X_train, y_train)
predicted = clf.predict(X_test)

np.mean(predicted == y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(predicted, y_test)
# # import matplotlib.pyplot as plt
# # fig = plt.sub_plot(1,1)
# # sns.countplot(y_test)

# # plt.sub_plot(1,2)
sns.countplot(predicted)

f, axes = plt.subplots(1, 1,  sharex=True)
sns.countplot(y_test)
