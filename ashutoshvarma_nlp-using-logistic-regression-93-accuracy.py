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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.preprocessing import LabelBinarizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize,sent_tokenize

from bs4 import BeautifulSoup

import re,string,unicodedata

from keras.preprocessing import text, sequence

from nltk.tokenize.toktok import ToktokTokenizer

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

from nltk import pos_tag

from nltk.corpus import wordnet

import keras

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,Dropout,CuDNNLSTM,GlobalMaxPool1D,Bidirectional
train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

submission = pd.read_csv("../input/quora-insincere-questions-classification/sample_submission.csv")
id = test['qid']
print('There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1]))

print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))
x=train.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
train.head()
train.rename(columns={'question_text' : 'text'},inplace = True)
train['target'].value_counts()
train.isnull().sum()
train.drop('qid',inplace = True,axis = 1)
train.head()
test.head()

test.rename(columns={'question_text' : 'text'},inplace = True)
test.isnull().sum()
test.drop('qid',inplace = True,axis = 1)
test.head()
train.head()
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()



#Removing the square brackets

def remove_between_square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)

# Removing URL's

def remove_urls(text):

    return re.sub(r'http\S+', '', text)

# Removing hashtags

def remove_hash(text):

    text = " ".join(word.strip() for word in re.split('#|_', text))

    return text

#Removing the stopwords from text

def remove_stopwords(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            final_text.append(i.strip())

    return " ".join(final_text)

#Removing the noisy text

def denoise_text(text):

    text = strip_html(text)

    text = remove_between_square_brackets(text)

    text = remove_urls(text)

    text = remove_hash(text)

    text = remove_stopwords(text)

    return text
train['text']=train['text'].apply(denoise_text)

test['text']=test['text'].apply(denoise_text)
max_features = 20000

maxlen = 100
tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(train.text)

tokenized_train = tokenizer.texts_to_sequences(train.text)

X = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(test.text)

sub_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
X.shape
sub_test.shape
from sklearn.model_selection import train_test_split

X_train , X_test ,y_train ,y_test = train_test_split(X,train['target'])
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(max_iter = 1000)

logit.fit(X_train,y_train)

score = logit.score(X_train,y_train)
score
y = logit.predict(X_test)
from sklearn.metrics import accuracy_score

print("accuracy of X_test data is : ",accuracy_score(y_test,y))
#classification report

from sklearn.metrics import classification_report

print("classification_report of X_test data is : ",classification_report(y_test,y))
final = logit.predict(sub_test)
final.shape
submission = pd.DataFrame({

        "qid": id,

        "prediction": final

    })
submission.to_csv("samplesubmission.csv", index=False)
sub = pd.read_csv('samplesubmission.csv')
sub.head()