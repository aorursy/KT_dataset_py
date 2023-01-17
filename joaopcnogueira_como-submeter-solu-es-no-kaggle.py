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
train_df = pd.read_csv('/kaggle/input/labdata-churn-challenge-2020/train.csv')

test_df  = pd.read_csv('/kaggle/input/labdata-churn-challenge-2020/test.csv')
train_df.head()
train_df = train_df.dropna(axis=1)

test_df  = test_df.dropna(axis=1)
train_df.drop('TotalCharges', axis=1, inplace=True)

test_df.drop('TotalCharges', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder



for cat_var in train_df.select_dtypes(include='O').columns:

    le = LabelEncoder()

    le.fit(train_df[cat_var])

    train_df[cat_var + '_num'] = le.transform(train_df[cat_var])

    train_df.drop(cat_var, axis=1, inplace=True)

    test_df[cat_var + '_num'] = le.transform(test_df[cat_var])

    test_df.drop(cat_var, axis=1, inplace=True)
train_df.head()
test_df.head()
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(max_depth=5, random_state=123)
X_train = train_df.drop('Churn', axis=1)

y_train = train_df['Churn']
dt.fit(X_train, y_train)
my_predictions = dt.predict(test_df)
submission = pd.read_csv('/kaggle/input/labdata-churn-challenge-2020/sample_submission.csv')
submission.head()
submission['Churn'] = my_predictions
submission.head()
submission.to_csv('submission_example.csv', index=False)