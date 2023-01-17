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
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
Y = train_data['Survived']

features = ['Pclass','Sex','SibSp','Parch']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

X.head()

Y.head()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X,Y)
prediction = model.predict(X_test)

prediction
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)