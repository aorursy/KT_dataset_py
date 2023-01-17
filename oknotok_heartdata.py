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



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.metrics import classification_report
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.age = df.age.astype(int)

df.head()
X = df.drop(["DEATH_EVENT"], 1)

y = df[["DEATH_EVENT"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train.values.ravel())

pred_train = rfc.predict(X_train)

report = classification_report(y_train, pred_train)

print(report)
model = RandomForestClassifier()

model.fit(X_test, y_test.values.ravel())

pred_test = model.predict(X_test)

report = classification_report(y_test, pred_test)

print(report)