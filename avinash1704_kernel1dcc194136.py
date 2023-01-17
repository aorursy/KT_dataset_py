import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input/jigsaw-toxic-comment-classification-challenge/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

test_data = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
train_data.head()
test_data.head()
print(train_data.shape)

print(test_data.shape)
train_data.describe()
train_data.info()
train_data.isna().sum()
import string

import re

import nltk

stopwords = nltk.corpus.stopwords.words("english")

from nltk.stem import PorterStemmer

ps = PorterStemmer()
def remove_punc(text):

    word = "".join([char.lower() for char in text if char not in string.punctuation])

    return word



train_data["removed_punch"] = train_data['comment_text'].apply(lambda x : remove_punc(x))

train_data.head()
def token(text):

    word = re.split("\W+",text)

    return word

train_data["token_word"] = train_data['removed_punch'].apply(lambda x : token(x))

train_data.head()
from sklearn.feature_extraction.text import TfidfVectorizer
word_vector = TfidfVectorizer(tokenizer=token,analyzer='word',max_features=1000)
train_vectorization = word_vector.fit_transform(train_data['comment_text'])

test_vectorization = word_vector.fit_transform(test_data['comment_text'])
train_vectorization.shape
test_vectorization.shape
# Creating DataFrame 

train_vectorization_df = pd.DataFrame(train_vectorization.toarray(), columns=word_vector.get_feature_names())

test_vectorization_df = pd.DataFrame(test_vectorization.toarray(), columns=word_vector.get_feature_names())
y_train = train_data[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

X_train = train_vectorization_df

X_test = test_vectorization_df
from sklearn.linear_model import LogisticRegression
target_label = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
predicted = np.zeros((X_test.shape[0],y_train.shape[1]))

predicted
for i,label in enumerate(target_label):

    lr = LogisticRegression(C=2,random_state = i,class_weight = 'balanced')

    print('Building {} model for column:{''}'.format(i,label)) 

    lr.fit(X_train,y_train[label])
from sklearn.metrics import classification_report

label = 'insult'

y_pred = lr.predict(X_train)

print(classification_report(y_train[label],y_pred))
for i in target_label:

    print(" Lable ",i,classification_report(y_train[i],y_pred))
y_predicted_labels = lr.predict_proba(X_train)[:,1]

y_predicted_labels
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_train['insult'], y_predicted_labels)

metrics.auc(fpr, tpr)
test_predicted = pd.DataFrame(predicted,columns=y_train.columns)

submission = pd.concat([test_data['id'],test_predicted],axis=1)

submission.to_csv('submit.csv',index=False)