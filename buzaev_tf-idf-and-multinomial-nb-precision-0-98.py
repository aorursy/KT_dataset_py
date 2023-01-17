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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
df = pd.read_csv("/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")
# look dataframe

df.head()
fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
del df["author_flair_text"]

del df["removed_by"]

del df["total_awards_received"]

del df["awarders"]

del df["created_utc"]

del df["full_link"]

del df["id"]
df.head()
df.info()
df.title.fillna(" ",inplace = True)
print(df["over_18"].value_counts())

sns.barplot(df["over_18"].value_counts().index, df["over_18"].value_counts().values)

print(round(df["over_18"][df["over_18"] == True].shape[0]/len(df), 2))
import string

# realization preprocessing

def preprocess(doc):

    # lower the text

    doc = doc.lower()

    # remove punctuation, spaces, etc.

    for p in string.punctuation + string.whitespace:

        doc = doc.replace(p, ' ')

    # remove extra spaces, merge back

    doc = doc.strip()

    doc = ' '.join([w for w in doc.split(' ') if w != ''])

    return doc
for colname in df.select_dtypes(include= np.object).columns:

    df[colname] = df[colname].map(preprocess)

df.head()
df['over_18'] = df['over_18'].map({True: 1, False: 0}).values
df.head()
lenTrue = int(1/3*df.over_18.value_counts().values[0])

lenTrue
df.over_18.value_counts()
df2 = df[df['over_18'] == 1].sample(n = 60982, replace=True)
df = df.append(df2, ignore_index=True)
df.head()
df.over_18.value_counts()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
y = df['over_18'].map({True: 1, False: 0}).values

y
df.drop(['over_18'], axis = 1, inplace=True)

df.head()
X = df
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
X_train
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS, ngram_range=(1, 2)).fit(df['title'])



X_train_vectors = vectorizer.transform(X_train['title'])

X_test_vectors = vectorizer.transform(X_test['title'])
X_train_vectors.shape, X_test_vectors.shape
num = 65

X_train_vectors[num].data
vectorizer.inverse_transform(X_train_vectors[num])[0][np.argsort(X_train_vectors[num].data)]
knn = KNeighborsClassifier().fit(X_train_vectors, y_train)
predicts = knn.predict((X_test_vectors))

print(classification_report(y_test, predicts))
clf = MultinomialNB().fit(X_train_vectors, y_train)
predicts = clf.predict((X_test_vectors))

print(classification_report(y_test, predicts))