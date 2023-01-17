# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import nltk

%matplotlib inline
df = pd.read_csv('../input/spam.csv',encoding = "ISO-8859-1")

df.dropna(inplace=True,axis=1)

df['tokens'] = df['v2'].apply(lambda x: nltk.word_tokenize(x))

df['pos_tags'] = df['tokens'].apply(lambda x:[t for w, t in nltk.pos_tag(x)])

df['pos_tag_sentence'] = df['pos_tags'].apply(lambda x: ' '.join(x))

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['v1'] = le.fit_transform(df['v1'])

X = df['pos_tag_sentence']

y = df['v1']

X_trainData, X_testData, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer(min_df=1)

vectorizer =TfidfVectorizer(sublinear_tf=True, max_df=0.5,

                                 stop_words='english')

X_train= vectorizer.fit_transform(X_trainData)

X_test = vectorizer.transform(X_testData)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

clf = RandomForestClassifier()

clf.fit(X_train.todense(),y_train)

y_pred = clf.predict(X_test.todense())

target_names = ['ham', 'spam']

print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import accuracy_score

print('accuracy Score: ', accuracy_score(y_test,y_pred))   

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train.todense(),y_train)

y_pred = clf.predict(X_test.todense())

target_names = ['ham', 'spam']

print(classification_report(y_test, y_pred, target_names=target_names))

print('accuracy Score: ', accuracy_score(y_test,y_pred)) 