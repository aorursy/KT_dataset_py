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
train=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
train.head()

test.head()
def punc(df):

    df['tweet'] = df['tweet'].str.replace('[#]','')

    print(df)
punc(train)

punc(test)
import nltk

from nltk.tokenize import TweetTokenizer

def tokenizer(df):

    tknzr = TweetTokenizer(strip_handles=True)

    df['tweet']= df['tweet'].apply(lambda x: tknzr.tokenize(x))

    print(df)

    
tokenizer(test)

tokenizer(train)
import nltk

from nltk.corpus import stopwords

stop=stopwords.words("english")

def stop_words(df):

    df['tweet']=df['tweet'].apply(lambda x: [i.lower() for i in x if i not in stop])

    print(df)
stop_words(train)

stop_words(test)
import re

def clean(df):

    df['tweet']=df['tweet'].apply(lambda x: [i for i in x if not re.match('[^\w\s]',i) and len(i)>3])

    print(df)
clean(train)

clean(test)
from nltk.stem import PorterStemmer

from textblob import Word

st = PorterStemmer()

def stemnlemm(df):

    df['tweet']=df['tweet'].apply(lambda x: [Word(st.stem(i)).lemmatize() for i in x])

    print(df)
stemnlemm(train)

stemnlemm(test)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500, analyzer='word', lowercase=False) 
train['tweet']=train['tweet'].apply(lambda x: " ".join(x) )

test['tweet']=test['tweet'].apply(lambda x: " ".join(x) )



X_train = cv.fit_transform(train['tweet'])
X_train
Y_train=pd.DataFrame(train['label'])

Y_train.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)
x_test.shape
from sklearn.naive_bayes import MultinomialNB

z=MultinomialNB()

z.fit(x_train,y_train)



pred=z.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import accuracy_score

cm=confusion_matrix(y_test,pred)

print(cm)

score = accuracy_score( y_test, pred)

print(score)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(x_train, y_train)



#predicting test results

pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,pred)

print(cm)

score = accuracy_score( y_test, pred)

print(score)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier( n_estimators=100 ,criterion='entropy', random_state=0)

classifier.fit(x_train, y_train)



#predicting test results

pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,pred)

print(cm)

score = accuracy_score( y_test, pred)

print(score)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)



#predicting test results

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,pred)

print(cm)

score = accuracy_score( y_test, pred)

print(score)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,y_pred)

print(cm)

score = accuracy_score( y_test, y_pred)

print(score)
from sklearn.svm import SVC

classifier= SVC(kernel='rbf',random_state=0)

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print(cm)

score = accuracy_score( y_test, y_pred)

print(score)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D

from keras.layers import Bidirectional
max_features = 220

tokenizer = Tokenizer(num_words = max_features, split = (' '))

tokenizer.fit_on_texts(train['tweet'].values)

X = tokenizer.texts_to_sequences(train['tweet'].values)

# making all the tokens into same sizes using padding.

X = pad_sequences(X, maxlen = max_features)

X.shape
Y = train['label'].values

model = Sequential()

model.add(Embedding(max_features, 64, input_length = X.shape[1], trainable=False))

model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X, Y,batch_size=1500,epochs = 5)
prediction = model.predict_classes(X)
from sklearn.metrics import accuracy_score

score = accuracy_score(Y, prediction)

print(score)
import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer



sid = SentimentIntensityAnalyzer()
train['score']=train['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

train.head()

train['compound']  = train['score'].apply(lambda score_dict: score_dict['compound'])

train.head()
train['comp_score'] = train['compound'].apply(lambda c: 1 if c >0 else 0)



train.head()
from sklearn.metrics import accuracy_score

score = accuracy_score(train['label'], train['comp_score'])

print(score)