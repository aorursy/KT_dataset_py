import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
train.head()
test.head()
train.describe()
train = train.drop(["Id"], axis = 1)



test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_val, y_train, y_val = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.2, random_state = 42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
from sklearn.metrics import classification_report, accuracy_score
model.score(X_train, y_train)
predictions = model.predict(X_val)

accuracy_score(y_val, predictions)
test.head()
test_pred = model.predict(test)
# Save test predictions to file

output = pd.DataFrame({'ID': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('submission.csv', index=False)