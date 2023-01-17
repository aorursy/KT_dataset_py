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
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
cols = ['id', 'keyword', 'location']
train.drop(cols, axis=1, inplace=True)
test.drop(cols, axis=1, inplace=True)
print(train.info())
print('\n')
print(test.info())
import string
from nltk.corpus import stopwords
def preprocess_text(text):
    #Convert to lower case
    #Remove Punctuations
    
    text = text.lower()
    
    text = [w for w in text if w not in string.punctuation]
   
    return ''.join(text)
train['text'] = train['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)
test.head()
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([('CountVectorizer' , CountVectorizer(analyzer=preprocess_text)),
                    ('tfidf' , TfidfTransformer()),
                    ('model' , RandomForestClassifier(criterion='entropy'))])
x_train = train['text']
y_train = train['target']

x_test = test['text']
pipeline.fit(x_train, y_train)
pred = pipeline.predict(x_test)
pred
test.head()
pipeline.predict(['apocalypse lighting spokane wildfires'])  # 4th column from test data
#The model is giving impressive answers

#The project was successful