# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None  # default='warn'



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder

import re
train_data = pd.read_csv("/kaggle/input/mini-flight-delay-prediction/flight_delays_train.csv")

train_data.head()
train_data.info()
test = pd.read_csv("/kaggle/input/mini-flight-delay-prediction/flight_delays_test.csv")

test.head()
test.info()
def monthDay2number(text):

    text = int(re.sub(r"c-", "", text))

    return text
train_data.Month = train_data.Month.map(lambda month: monthDay2number(month))

train_data.DayofMonth = train_data.DayofMonth.map(lambda day: monthDay2number(day))

train_data.DayOfWeek = train_data.DayOfWeek.map(lambda day: monthDay2number(day))

train_data.drop(columns=['Origin','Dest'], inplace=True)



train_data.dep_delayed_15min = train_data.dep_delayed_15min.map(lambda label: 0 if label=='Y' else 1)
test.Month = test.Month.map(lambda month: monthDay2number(month))

test.DayofMonth = test.DayofMonth.map(lambda day: monthDay2number(day))

test.DayOfWeek = test.DayOfWeek.map(lambda day: monthDay2number(day))

test.drop(columns=['Origin','Dest'], inplace=True)
train_data = pd.get_dummies(train_data)

test = pd.get_dummies(test)



for column in train_data.columns:

    if not column in test.columns:

        test[column] = pd.DataFrame(0, index=np.arange(len(test)), columns=[column])



for column in test.columns:

    if not column in train_data.columns:

        train_data[column] = pd.DataFrame(0, index=np.arange(len(train_data)), columns=[column])
train, test = train_test_split(train_data)
labels_train = train.dep_delayed_15min

train.drop(columns="dep_delayed_15min", inplace=True)



labels_test = test.dep_delayed_15min

test.drop(columns="dep_delayed_15min", inplace=True)
clf_svm = svm.SVC(class_weight='balanced')

clf_svm.fit(train, labels_train)
labels_predicted = clf_svm.predict(test)

print('F1-score: ', metrics.f1_score(labels_test, labels_predicted))
metrics.confusion_matrix(labels_test, labels_predicted)
clf_randomforest = RandomForestClassifier(n_estimators=100, class_weight='balanced')

clf_randomforest.fit(train, labels_train)
labels_predicted = clf_randomforest.predict(test)

print('F1-score: ', metrics.f1_score(labels_test, labels_predicted))
metrics.confusion_matrix(labels_test, labels_predicted)
clf_boosting =  GradientBoostingClassifier()

clf_boosting.fit(train, labels_train)
labels_predicted = clf_boosting.predict(test)

print('F1-score: ', metrics.f1_score(labels_test, labels_predicted))
metrics.confusion_matrix(labels_test, labels_predicted)
clf_mlp = MLPClassifier(hidden_layer_sizes=(500, 20), max_iter=3000)

clf_mlp.fit(train, labels_train)
labels_predicted = clf_mlp.predict(test)

print('F1-score: ', metrics.f1_score(labels_test, labels_predicted))
metrics.confusion_matrix(labels_test, labels_predicted)
labels_train_data = train_data.dep_delayed_15min

train_data.drop(columns="dep_delayed_15min", inplace=True)



clf_boosting.fit(train_data, labels_train_data)
labels_predicted = clf_boosting.predict(test)
labels_submission = pd.Series(labels_predicted)

labels_submission = labels_submission.map(lambda label: 'Y' if label==0 else 'N')

labels_submission.to_csv('submission.csv', index=False)