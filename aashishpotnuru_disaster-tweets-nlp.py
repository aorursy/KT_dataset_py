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
import numpy as np
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from nltk.corpus import stopwords
import re
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train.head()
stop_words = set(stopwords.words('english'))
def data_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        train[col][ind] = string
for index, row in train.iterrows():
    if type(row['text']) is str:
        data_text_preprocess(row['text'], index, 'text')
train.head()
X = train["text"]
y = train["target"]
from sklearn.model_selection import train_test_split
X_train, test_df, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
X_train.shape , test_df.shape
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit_transform(X_train)
simple_train_dtm = vect.transform(X_train)
# examine the vocabulary and document-term matrix together
X_train = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
X_train.shape
test = vect.transform(test_df)
X_test = pd.DataFrame(test.toarray(), columns=vect.get_feature_names())
X_test.shape
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test.shape
def data_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        train[col][ind] = string
for index, row in test.iterrows():
    if type(row['text']) is str:
        data_text_preprocess(row['text'], index, 'text')
test = vect.transform(test["text"])
# examine the vocabulary and document-term matrix together
Xtest = pd.DataFrame(test.toarray(), columns=vect.get_feature_names())
Xtest.shape
y_pred = clf.predict(Xtest)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = y_pred
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)