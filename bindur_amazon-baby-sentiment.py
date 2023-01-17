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
df = pd.read_csv('../input/amazon-baby-sentiment-analysis/amazon_baby.csv')
df
def sentiment(x):
    if x>3: 
        return 2
    elif x==3:
        return 1
    else:
        return 0


df['sentiment'] = df['rating'].apply(lambda x: sentiment(x) )
df
df.isna().sum()
df = df[df['name'].notnull()]
df = df[df['review'].notnull()]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
x=df['review']
y=df['sentiment']
X_train,X_test,y_train,y_test=train_test_split(x,y)
vect= CountVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)
accuracy_score(logreg.predict(X_test),y_test)
nb = MultinomialNB()
nb.fit(X_train,y_train)
accuracy_score(nb.predict(X_test),y_test)
