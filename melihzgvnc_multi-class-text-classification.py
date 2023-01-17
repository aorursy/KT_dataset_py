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
df = pd.read_csv('../input/medium-post-titles/medium_post_titles.csv')

df
df.info()
df.isna().sum() 
df.drop(df[df.subtitle.isna()].index, inplace=True) #Rows with missing values are cleared

len(df)
df.reset_index(drop=True, inplace=True)

df
df.describe()
df.title[0]+" "+df.subtitle[0]
df["text"] = ""

for row in range(len(df)):

    df["text"][row] = df.title[row]+" "+df.subtitle[row]

df    
df.drop(columns=['title','subtitle','subtitle_truncated_flag'], inplace=True)
import re

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



def preprocessing(text):

    text = re.sub(r'\W+',' ',text)

    text = re.sub(r'\s+',' ',text)    

    text = text.lower()

    result = ''

    for word in word_tokenize(text):

        if word not in list(stopwords.words('english')):

            result += word + ' '

            

    return result
df["text"] = df.text.apply(preprocessing)
df
df.category.unique()
len(df.category.unique())
import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize=(18,20))

plt.barh(df.category.unique(), df.category.value_counts(), height=0.75, align='center')

plt.xlabel('Medium Post Counts by Category')

plt.ylabel('Categories')

plt.show()
from sklearn.model_selection import train_test_split



X = df.text

y = df.category



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', LogisticRegression(n_jobs=1, C=1e5))

])





model.fit(X_train,y_train)

y_pred = model.predict(X_test)



print('accuracy: ', accuracy_score(y_pred, y_test))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import classification_report, accuracy_score



model = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=0.001))

])





model.fit(X_train,y_train)

y_pred = model.predict(X_test)



print('accuracy: ', accuracy_score(y_pred, y_test))