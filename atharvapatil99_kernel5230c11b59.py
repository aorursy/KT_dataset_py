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
# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
path = '/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'

df = pd.read_csv(path)
df.head()
df['length'] = df['review'].apply(len)
df.head()
sns.set_style('whitegrid')

g = sns.FacetGrid(df, col='sentiment', size=6)

g.map(plt.hist, 'length', bins=70)

plt.figure(figsize=(10,8))

sns.boxplot(x='sentiment', y='length', data=df)
df.describe()
df[df['length']==13704]
X = df['review']

y = df['sentiment']
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)

pred = nb.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))