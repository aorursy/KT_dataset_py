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
from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head()
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df.head()
train_df.info()
y = train_df.Survived
features = ['Pclass', 'Sex', 'SibSp', 'Parch']

x = pd.get_dummies(train_df[features])
x_test = pd.get_dummies(test_df[features])
clf = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

clf.fit(x, y)

predict = clf.predict(x_test)
output = pd.DataFrame({'PassengerId' : test_df.PassengerId, 'Survived' : predict})

output.to_csv('my_submission.csv', index = False)

print('Submission saved')