import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

import re
#import data

train_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
#Taining data

train_data.head(10)
test_data.head(5)
target_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
#check for any null comments

null_comments = train_data[train_data['comment_text'].isnull()]

print("Number of null comments: " + str(len(null_comments)))
#Counting total comments with above labels

print(train_data[target_labels].sum())
#Plot for text length in each comments

train_data['char_length'] = train_data['comment_text'].apply(lambda x:len(str(x)))

sns.set()

train_data['char_length'].hist()

plt.show()
#cleaning up comment text for every comment in train and test data

#function for cleaning up comments

def clean_comment(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text

    
train_data['comment_text'] = train_data['comment_text'].apply(lambda x:clean_comment(x))

test_data['comment_text'] = test_data['comment_text'].apply(lambda x:clean_comment(x))
#vectorizing the text data

vec = TfidfVectorizer(max_features = 5000,stop_words = 'english')

x_train = vec.fit_transform(train_data['comment_text'])

x_test = vec.transform(test_data['comment_text'])
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

classifier = LogisticRegression(C=12.0)

for label in target_labels:

    print('Fitting the model for label: ' + label)

    y_labels = train_data[label]

    classifier.fit(x_train,y_labels.values)

    y_pred = classifier.predict(x_train)

    print('Training accuracy: ' + str(accuracy_score(y_labels,y_pred)))

    y_test_prob = classifier.predict(x_test)

    submission[label] = y_test_prob
#create submission file

submission.to_csv('test_submissions.csv',index = False)