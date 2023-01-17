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
import numpy as np 

import pandas as pd 

import os

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
X = train["text"]

y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cvect = CountVectorizer(stop_words = 'english')

X_train_cv = cvect.fit_transform(X_train)

X_test_cv = cvect.transform(X_test)
model  = MultinomialNB()

model.fit(X_train_cv, y_train)

pred = model.predict(X_test_cv)
pred
accuracy_score(y_test,pred)
sub_test = test["text"]

sub_test_cv = cvect.transform(sub_test)

sub_preds = model.predict(sub_test_cv)
sub["target"] = sub_preds

sub.to_csv("submission.csv",index=False)