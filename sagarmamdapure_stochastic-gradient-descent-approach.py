# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
input_data = pd.read_csv('../input/SPAM text message 20170820 - Data.csv')

input_data.head()

input_data['Num_label'] = input_data.Category.map({'ham' : 0, 'spam' : 1})

X = input_data.Message
y = input_data.Num_label
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = True)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

X_train_sparse = vect.fit_transform(X_train)
X_test_sparse = vect.transform(X_test)

from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier()

classifier.fit(X_train_sparse, y_train)

predicted_score = classifier.predict(X_test_sparse)

from sklearn import metrics

metrics.accuracy_score(y_test, predicted_score)*100