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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
features = ['Pclass', 'Sex']



X = pd.get_dummies(train_data[features])

y = train_data['Survived'].values
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=5, stratify=y)





knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
knn.score(X_test,y_test)
X_test_r = pd.get_dummies(test_data[features])
predictions = knn.predict(X_test_r)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)