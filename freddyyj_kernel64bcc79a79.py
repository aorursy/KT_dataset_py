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

train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train=train.drop(columns='textID')
test=test.drop(columns='textID')
train.head()
train['text']=train['text'].fillna(" ")
test['text']=test['text'].fillna(" ")
train['selected_text']=train['selected_text'].fillna(" ")
from sklearn.preprocessing import LabelEncoder

# negative: 0, neutral: 1, positive: 2
cat_features = ['sentiment']
encoder = LabelEncoder()

# Apply the label encoder to each column
encoded = train[cat_features].apply(encoder.fit_transform)


train=train.drop(columns='sentiment')
train=train.join(encoded)
train.head()

encoded = test[cat_features].apply(encoder.fit_transform)


test=test.drop(columns='sentiment')
test=test.join(encoded)
train.head()
x_train=train.drop(columns='sentiment')
y_train=train['sentiment']
x_test=test.drop(columns='sentiment')
y_test=test['sentiment']
for each in [train, test]:
    print(f"sentiment fraction = {each.sentiment.mean():.4f}")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
# Original code by https://www.kaggle.com/vanshjatana/text-classification-from-scratch
# x_train,x_test,y_train,y_test = train_test_split(data['text'], data.true, test_size=0.2, random_state=2020)
x_train=train['text']
x_test=test['text']

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LinearSVC())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
print(confusion_matrix(y_test, prediction))