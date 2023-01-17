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
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df2 = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df3 = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df.head()
df.info()
df.drop(['id','keyword','location'],axis=1,inplace=True)

df2.drop(['id','keyword','location'],axis=1,inplace=True)
df.head()
df2.head()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer()
from sklearn.model_selection import train_test_split
X = df['text']

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb = MultinomialNB()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
nb = MultinomialNB()
cv = CountVectorizer()
X = cv.fit_transform(X)
nb.fit(X,y)
df2
df2 = cv.transform(df2['text'])
df2
pred = nb.predict(df2)
df3.head()
df3['target'] = pred
df3['target'].value_counts()
df3
df3.to_csv('predict.csv',line_terminator='\r\n',index=False)