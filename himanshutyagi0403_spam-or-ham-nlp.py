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
df = pd.read_csv('../input/spam.csv',encoding='latin1')
df.head()
df = df.iloc[:,0:2]
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for text in df.v2:
    text = re.sub('[^a-zA-Z]',' ', text)
    text = text.split()
    text = [words for words in text if words not in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()
y = df.v1
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, random_state=0)
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(x_train,y_train)

y_pred = classifier_lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test,y_pred)
acc

cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=100)
classifier_rf.fit(x_train,y_train)

y_pred_rf = classifier_rf.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc_rf = accuracy_score(y_test,y_pred_rf)
acc_rf

cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf
acc_rf

