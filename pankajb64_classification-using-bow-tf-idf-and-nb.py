# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

# Any results you write to the current directory are saved as output.

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
table = pd.read_csv('../input/spam.csv', encoding='latin-1')

table.head()
x_train, x_test, y_train, y_test = train_test_split(table.v2, table.v1, test_size=0.15, random_state=30)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)



print(x_train[:5])

print(y_train[:5])
vect = TfidfVectorizer()

x_train_tf = vect.fit_transform(x_train)

x_test_tf = vect.transform(x_test)



x_train_tf = x_train_tf.toarray()

x_test_tf = x_test_tf.toarray()



print(x_train_tf.shape)

print(x_test_tf.shape)



print(vect.vocabulary_)

print(x_train_tf[:5])
nb_classifier = GaussianNB()

nb_classifier.fit(x_train_tf, y_train)

y_predict = nb_classifier.predict(x_test_tf)
print(accuracy_score(y_test, y_predict))