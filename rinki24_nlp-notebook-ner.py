# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
df = pd.read_csv('/kaggle/input/ner_dataset.csv', encoding = "ISO-8859-1")

df = df[:100000]

df.head()
df.isnull().sum()
df = df.fillna(method='ffill')

df['Sentence #'].nunique(), df.Word.nunique(), df.Tag.nunique()
df.head()
df.groupby('Tag').size().reset_index(name='counts')
X = df.drop('Tag', axis=1)

X.head()
X.to_dict('records')
v = DictVectorizer(sparse=False)

X = v.fit_transform(X.to_dict('records'))


y = df.Tag.values

classes = np.unique(y)

classes = classes.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)

X_train.shape, y_train.shape
per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)

per.partial_fit(X_train, y_train, classes)
# We will just pop out class o

new_classes = classes.copy()

new_classes.pop()

new_classes
print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))
sgd = SGDClassifier()

sgd.partial_fit(X_train, y_train, classes)

print(classification_report(y_pred=sgd.predict(X_test), y_true=y_test, labels=new_classes))
nb = MultinomialNB(alpha=0.01)

nb.partial_fit(X_train, y_train, classes)

print(classification_report(y_pred=nb.predict(X_test), y_true=y_test, labels = new_classes))