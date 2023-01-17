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
import matplotlib.pyplot as plt
import seaborn as sns
import string
train = pd.read_csv('/kaggle/input/train.csv',encoding='utf8')
test = pd.read_csv('/kaggle/input/test.csv',encoding='utf8')
train.head()
train.isnull().sum()
test.isnull().sum()
train=train.drop(['id','keyword','location'],axis=1)
test=test.drop(['keyword','location'],axis=1)
test.head()
sns.heatmap(train.isnull(),yticklabels = False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels = False,cmap='viridis')
def remove_punc(text):
    # Dealing with Punctuation
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
import re
train['text']=train['text'].apply(lambda x : remove_punc(x))
test['text']=test['text'].apply(lambda x : remove_punc(x))
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
def data_cleaner(text):        
    lower_case = text.lower()
    tokens=word_tokenize(lower_case)
    return (" ".join(tokens)).strip()

def remove_stopwords (text):        
    list1=[word for word in text.split() if word not in stopwords.words('english')]
    return " ".join(list1)
train['text'] = train['text'].apply(lambda x : data_cleaner(x))
test['text'] = test['text'].apply(lambda x : data_cleaner(x))

train['text'] = train['text'].apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x :remove_stopwords(x))
X = train['text']
y = train['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state = 101)
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
def prediction(pipeline, x_train, y_train,testtext):
    
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(testtext)
    return y_pred
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', RidgeClassifier())
        ])
checker_pipeline
vectorizer.set_params(stop_words=None, max_features=10000, ngram_range=(1,4))
prediction=prediction(checker_pipeline,train['text'],train['target'],test['text'])
submission = pd.DataFrame({
    "id":test['id'],
    "target":(prediction > 0.5).astype(int)})
submission.to_csv("submission.csv", index=False, header=True)