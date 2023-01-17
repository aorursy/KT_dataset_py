# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import re

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read dataset

df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



# first five rows

df_train.head()
# describe

df_train.describe()
# Function for removing Punctuation from the text

def remove_punct(text):

  # Remove punctuation and digit from text

  return "".join([t for t in text if t not in string.punctuation and not t.isdigit()]) # isdigit():- select only digit isaplha():- select only alphabet

# Function for removing URL from the text

def remove_url(text):

  return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE)
# Apply First URL removal then punctuation removal 

df_train["no_url"] = df_train['text'].apply(lambda x: remove_url(str(x)))

df_train["no_punct"] = df_train['no_url'].apply(lambda x: remove_punct(str(x)).lower()) #lower() convert the text to lower case
df_train.sample(10)
# Assign variable `source` for text and `label` for  target

source = df_train['no_punct'].values

label = df_train['target'].values

# split the data into train and validation set

text_train, text_test, y_train, y_test = train_test_split(source, label, test_size=0.25, random_state=1000)
# Define CountVectorizer() object

#vectorizer = CountVectorizer()

# Define CountVectorizer() object

vectorizer = TfidfVectorizer()

vectorizer.fit(text_train)
# Conver the text data into vector form

X_train = vectorizer.transform(text_train)

X_test  = vectorizer.transform(text_test)
#print(X_train)
# Define LogisticRegression() object for model prepration

classifier = LogisticRegression()

# Fit the model

classifier.fit(X_train, y_train)
# Calculate score

score = classifier.score(X_test, y_test)

# Print the accuracy of the model

print("Accuracy:", score)
# Apply First URL removal then punctuation removal 

df_test["no_url"] = df_test['text'].apply(lambda x: remove_url(str(x)))

df_test["no_punct"] = df_test['no_url'].apply(lambda x: remove_punct(str(x)).lower()) #lower() convert the text to lower case

df_test.head()
# Assign variable `source_test` for text on test dataset

source_test = df_test['no_punct'].values

# conver to vector

source_vect  = vectorizer.transform(source_test)
predictions=classifier.predict(source_vect) # to get target column as either 0 or 1 value no probability

df_submission['target']=predictions

df_submission.to_csv('submission.csv', index=False)

df_submission.head()