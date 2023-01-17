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
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd
Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test_data = pd.read_csv("../input/Kannada-MNIST/test.csv")

train_data = pd.read_csv("../input/Kannada-MNIST/train.csv")
X = train_data.drop('label', axis=1)

Y = train_data['label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
%%time

linear = LogisticRegression(C=20, solver='lbfgs', multi_class='multinomial')

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)
%%time

linear = LogisticRegression(multi_class='multinomial', solver='lbfgs')

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)
%%time

linear = LogisticRegression(multi_class='multinomial', solver='newton-cg')

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)
%%time

linear = LogisticRegression(multi_class='multinomial', solver='sag')

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)
%%time

linear = LogisticRegression(multi_class='multinomial', solver='saga')

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)
%%time

linear = LogisticRegression(multi_class='multinomial', solver='saga', penalty='elasticnet', l1_ratio=0)

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)
%%time

linear = LogisticRegression()

linear.fit(x_train, y_train)

y_hat = linear.predict(x_test)

score = accuracy_score(y_test.values, y_hat.round())

print ("score", score)