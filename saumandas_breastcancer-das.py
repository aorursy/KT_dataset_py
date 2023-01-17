# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

input_file = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(input_file, test_size=0.2, random_state=42)

pipeline = Pipeline([("standard", StandardScaler())])

train_set.info()
test_set.info()
y_train = (train_set['diagnosis'] == 'M')

train_set.drop(["diagnosis","Unnamed: 32"] , axis = 1,inplace=True)

train_set.drop("id", axis=1, inplace=True)

y_train
y_test = (test_set['diagnosis'] == 'M').astype(np.uint8)

test_set.drop(["diagnosis", "Unnamed: 32"], axis = 1,inplace=True)

test_set.drop("id", axis=1, inplace=True)



y_test.head()
print(type(train_set))

train_set = pipeline.fit_transform(train_set)

test_set = pipeline.fit_transform(test_set)
from sklearn.svm import SVC

svc_clf = SVC(gamma="auto")

svc_clf.fit(train_set, y_train)
predictions = svc_clf.predict(train_set)
from sklearn import metrics

acc_score = metrics.accuracy_score(predictions, y_train)

acc_score
final_pred = svc_clf.predict(test_set)

acc_score = metrics.accuracy_score(final_pred, y_test)

acc_score
from sklearn.linear_model import LogisticRegression

log_class = LogisticRegression()

log_class.fit(train_set, y_train)

acc_score=metrics.accuracy_score(log_class.predict(train_set), y_train)

acc_score
log_predictions = log_class.predict(test_set)

acc_score=metrics.accuracy_score(log_predictions, y_test)

acc_score