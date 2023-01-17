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
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

sample_sub = pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')
train.head()
train.isnull().sum()
train_df = train.drop('id',1)
train_df = train_df.replace({'Male':1,'Female':2})
train_df.Vehicle_Age.value_counts()
train_df = pd.get_dummies(train_df)
train_df.head()
test.head()
test = test.drop('id',1)
test = test.replace({'Male':1,'Female':2})
test = pd.get_dummies(test)
test.head()
X = train_df.drop('Response',1)

y = train_df['Response']
from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.28,random_state=4)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
dtc = DecisionTreeClassifier()

rtc = RandomForestClassifier()

etc = ExtraTreesClassifier()
from sklearn.metrics import r2_score, confusion_matrix
dtc.fit(X_train,X_test)

r2_score(y_test,dtc.predict(y_train))
etc.fit(X_train,X_test)

r2_score(y_test,etc.predict(y_train))
from sklearn.tree import ExtraTreeRegressor

etc = ExtraTreeRegressor()

etc.fit(X_train,X_test)



r2_score(y_test,etc.predict(y_train))
y_pred = dtc.predict(test)



submission = pd.DataFrame({

        "id": sample_sub["id"],

        "Response": y_pred

    })



submission.to_csv('my_submission.csv', index=False)