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
import re



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
df = pd.read_csv('/kaggle/input/document-classification/file.txt')
df.head()
df.shape
df.isnull().sum()
def get_label(text):

     for i in text:

            return int(i[0])
df['label'] = df['5485'].apply(lambda x: get_label(x))
df.head()
df.columns = ('text', 'label')
df.head()
df['text']=df['text'].str[1:]
df.head()
df.tail()
df.label.value_counts().plot(kind='bar')
#remove special characters and punctuation

df['text'] = df['text'].replace(r'[^A-Za-z0-9 ]+', '')



#remove single letters from text

df['text'] = df['text'].apply (lambda x: re.sub(r"((?<=^)|(?<= )).((?=$)|(?= ))", '', x).strip())
df.sample(10)
vectorizer = CountVectorizer(stop_words='english')
X = df['text']

y = df['label']
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 88)
X_vect = vectorizer.fit_transform(X_train)
nb = MultinomialNB()
nb.fit(X_vect,y_train)
y_pred = nb.predict(vectorizer.transform(X_test))
print(accuracy_score(y_test,y_pred))