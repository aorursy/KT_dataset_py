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
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re

nltk.download('stopwords')
# nltk.download()
# Load the dataset

col_names = ["label", "messages"]
df = pd.read_csv("/kaggle/input/sms-spam-collection-data-set/SMSSpamCollection", names = col_names, sep='\t')
# Check for null values
df.isnull().sum()
# Apply lemmitization

def lemma(sentences):
    lemmatizer = WordNetLemmatizer()
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        sentences[i] = ' '.join(words)
    return sentences
# Apply Stopword Removal

def stemming(sentences):
    ps = PorterStemmer()
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
        sentences[i] = ' '.join(words)
    return sentences
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]',' ',df['messages'][i])
    review = review.lower()
    review = review.split()
    review = stemming(review)
    review = ' '.join(review)
    corpus.append(review)
corpus
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()
X

y = pd.get_dummies(df['label'])
y = y.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y , test_size = 0.20, random_state = 0)
y_test
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)
from sklearn.metrics import confusion_matrix 
confusion_m = confusion_matrix(y_test,y_pred)
confusion_m
from sklearn.metrics import accuracy_score
accu = accuracy_score(y_test,y_pred)
accu