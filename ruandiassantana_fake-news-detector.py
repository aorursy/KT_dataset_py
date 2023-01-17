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



true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

false = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
true.head()
false.head()
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import train_test_split
true['real'] = 1
true.head()
false['real'] = 0
false.head()
df = pd.concat((true,false),axis=0)
df.head()
df['ttext'] = df['title']+df['text']
df.drop(['title','text'],axis=1,inplace=True)
X = df['ttext']

y = df['real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pipeline = Pipeline([

    ('bow',CountVectorizer()),

    ('naive_bayes',MultinomialNB())

])
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))