# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from pathlib import Path

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df.head()
submi_df = df_test['PassengerId'].to_frame()

submi_df.head()
# df = pd.read_csv('../input/gender_submission.csv')

# df.head()
features = ['Pclass', 'SibSp', 'Parch']

X = df[features]

X_test = df_test[features]

y = df['Survived']

X.head()
lr = LogisticRegression()
lr.fit(X, y)

lr.coef_
y_pred = pd.DataFrame(lr.predict(X_test))

y_pred[:5]
submi_df['Survived'] = y_pred

submi_df.to_csv('myfirstsubmissiontokagglecomp.csv', index_label = False, index = False)
submi_df.head()