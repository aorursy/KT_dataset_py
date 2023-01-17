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
train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
sample_submittion = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")
test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
train
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
import string

def preprocess_text(text):
    text = text.lower()
    
    text = [w for w in text if w not in string.punctuation]
    
    return ''.join(text)
train['comment_text'] = train['comment_text'].apply(preprocess_text)
test['comment_text'] = test['comment_text'].apply(preprocess_text)
train
#Training Set
x_train = train['comment_text']
y_train = train.drop('comment_text', axis=1)

#Testing Set
x_test = test['comment_text']
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
model = Pipeline([('CV', CountVectorizer(analyzer=preprocess_text, stop_words='english')),
                 ('Tfidf', TfidfTransformer()),
                 ('forest', RandomForestClassifier(criterion='entropy'))])
model.fit(x_train, y_train)
x_test[12561]
model.predict(['i dont anonymously edit articles at all']) # 4th column from x_test
model.predict(['stop already your bullshit is not welcome here im no fool and if you think that kind of explination is enough well pity you'])
# 153163 column from x_test
model.predict(['sigh  why is this in mainspace    stupid bot  every one in a so many thousand you do this  and i dont see the reason'])
# 12561 column from x_test
