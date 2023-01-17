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
#importing libraries

import spacy

import nltk

from nltk.tokenize.toktok import ToktokTokenizer

# nlp = spacy.load('en_core', parse=True, tag=True, entity=True)

tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

stopword_list.remove('no')

stopword_list.remove('not')

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
#importing files

test_df=pd.read_csv('../input/nlp-getting-started/test.csv')

train_df=pd.read_csv('../input/nlp-getting-started/train.csv')

submission_df=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train_df.head()
test_df.head()
submission_df.head()
# all_df=pd.concat([train_df,test_df])

# all_df.head()

all_df=[train_df,test_df]

# all_df.drop(columns=['keyword','location'],inplace=True)

# all_df.head()

for df in all_df:

    df.drop(columns=['keyword','location'],inplace=True)

    

print(train_df.head())

print(test_df.head())
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])
# all_df['text']=all_df['text'].apply(lambda x : x.split())

# all_df.head()
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
submission_df["target"] = clf.predict(test_vectors)

submission_df.head()
submission_df.to_csv("submission.csv", index=False)