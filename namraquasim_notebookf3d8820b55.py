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
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
train=pd.read_csv("../input/nlp-getting-started/train.csv")
test=pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
train.tail()
train['target'].unique()
train[train['target']==1].head()
train[train['target']==0].head()
train[train['target']==1].count()
train[train['target']==0].count()
test.head()
test.tail()
submission=pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission.head()

train.head()
train.drop('keyword',axis=1,inplace=True)
train
train.drop('location',axis=1, inplace=True)
train.head()
def word_count(sent):
    return len(sent.split())
train['word_count'] = train.text.apply(word_count)
train.head()
def remove_urls(sent):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',sent)

train['text'] = train.text.apply(remove_urls)
train
import emoji
def emoji_to_text(sent):
    e_sent = emoji.demojize(sent)
    emo = re.findall(':(.*?):',e_sent)
    for e in emo:
        e_sent = e_sent.replace(':{}:'.format(e),'{}'.format(e))
    return e_sent

train['text'] = train['text'].apply(emoji_to_text)
def remove_hashtags(text):
    return re.sub('#\w*[a-zA-Z]\w*','',text)

train['text'] = train['text'].apply(remove_hashtags)
train
def replace_username(sent):
    usernames = re.findall('@[A-Za-z0-9_$]*',sent)
    for un in usernames:
        un = re.sub('@','',un)
        sent = sent.replace('@{}'.format(un),'{}'.format(un))
    return sent

train['text'] = train['text'].apply(replace_username)
train
def remove_number(text):
    return re.sub('#[0-9]+','',text)

train['text'] = train['text'].apply(remove_number)
train
def remove_punctuations(text):
    return re.sub('[.?"\'`\,\-\!:;\(\)\[\]\\/“”]+?','',text)

train['text'] = train['text'].apply(remove_punctuations)
train
def remove_symbols(text):
    return re.sub('[~:*ÛÓ_å¨È$#&%^ª|+-]+?','',text)

train['text'] = train['text'].apply(remove_symbols)
train
from sklearn.model_selection import train_test_split

d_train,d_test = train_test_split(train,test_size = 0.2,random_state=0)

print("train shape : ", d_train.shape)
print("valid shape : ", d_test.shape)
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

X_train = vectorizer.fit_transform(d_train.text.values)
X_test = vectorizer.transform(d_test.text.values)

y_train = d_train.target.values
y_test = d_test.target.values

print("X_train.shape : ", X_train.shape)
print("X_train.shape : ", X_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_valid.shape : ", y_test.shape)
from sklearn.linear_model import LogisticRegression

lreg = LogisticRegression()
lreg.fit(X_train,y_train)
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
lreg_prediction = lreg.predict(X_test)
lreg_accuracy = accuracy_score(y_test,lreg_prediction)
print("training accuracy Score    : ",lreg.score(X_train,y_train))
print("Validdation accuracy Score : ",lreg_accuracy )
def preprocessing(df):
    #df.drop('keyword',axis=1,inplace=True)
    #df.drop('location',axis=1, inplace=True)
    df['Char_length']       = df['text'].apply(len)
    df['word_count']        = df.text.apply(word_count)
    df['text']              = df.text.apply(remove_urls)
    df['text']              = df['text'].apply(emoji_to_text)
    df['text']              = df.text.apply(remove_hashtags)
    df['text']              = df.text.apply(replace_username)
    df['text']              = df.text.apply(remove_number)
    df['text']              = df.text.apply(remove_punctuations)
    df['text']              = df.text.apply(remove_symbols)
                                            
    return df
processed_test = preprocessing(test)

processed_test
def bagofword(df):
    X = vectorizer.transform(df.text.values)
    return X    

def predict_test(model,x):
    return model.predict(x)
X_test = bagofword(processed_test)
target = predict_test(lreg,X_test)
submission['target'] = target
submission.to_csv('submission.csv',index=False)
submission.head()
