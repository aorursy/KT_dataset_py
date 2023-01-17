# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

print('start read data')
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

print("Prepare text data of Train and Test ... ")
def generate_text(data):
    ingredients = data['ingredients']
    text_data = list()
    for doc in ingredients:
        str_arr = list()
        for s in doc:
            str_arr.append(s.replace(' ', ''))
        text_data.append(" ".join(str_arr).lower())
    # text_data = [" ".join(doc).lower() for doc in ingredients]
    return text_data

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc for doc in train['cuisine']]

# Feature Engineering
print("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)


def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float16')
    return x


X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")

print("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)

print("Train the model ... ")
lr = LogisticRegression(max_iter=500)
lr.fit(X, y)

# Predictions
print("Predict on test data ... ")
y_test = lr.predict(X_test)
y_predict = lb.inverse_transform(y_test)

print("Generate Submission File ... ")
test_id = [doc for doc in test['id']]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_predict}, columns=['id', 'cuisine'])
sub.to_csv('tfidf_lr_output2.csv', index=False)


