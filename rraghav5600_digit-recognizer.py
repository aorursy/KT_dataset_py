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
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X = df.drop(columns='label')

y = df['label']

t_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_x = t_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
model_SVC_scale = SVC(gamma='scale')

model_SVC_scale.fit(X_train, y_train)

pred = model_SVC_scale.predict(X_test)

accuracy_score(pred, y_test)
model_rf = RandomForestClassifier(random_state=0, n_estimators=300)

model_rf.fit(X_train, y_train)

pred = model_rf.predict(X_test)

accuracy_score(pred, y_test)
model_SVC_scale.fit(X,y)

test_y = model_SVC_scale.predict(test_x)

file = pd.DataFrame({'ImageId':[i+1 for i in range(len(t_df))], 'Label':test_y})

file.to_csv('submission_SVC_scale.csv', index = False)

file.head()
model_rf.fit(X,y)

test_y = model_rf.predict(test_x)

file = pd.DataFrame({'ImageId':[i+1 for i in range(len(t_df))], 'Label':test_y})

file.to_csv('submission_rf.csv', index = False)

file.head()