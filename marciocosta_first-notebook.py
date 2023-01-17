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

train.shape
train.nunique()
test.head()
test.shape
#nulls columns?

missing_val_count_by_column_test = (test.isnull().sum())

print(missing_val_count_by_column_test[missing_val_count_by_column_test > 0])
train.describe()
train = train.drop(["Id"], axis = 1)



test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.2)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#nodistancetoroad

X_train_no_road = X_train.drop(['Horizontal_Distance_To_Roadways'],axis=1)

X_valid_no_road = X_valid.drop(['Horizontal_Distance_To_Roadways'],axis=1)
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train_no_road, y_train)
from sklearn.metrics import classification_report, accuracy_score
model.score(X_train_no_road, y_train)
predictions = model.predict(X_valid_no_road)

accuracy_score(y_valid, predictions)
X_test_no_road = test.drop(['Horizontal_Distance_To_Roadways'],axis=1)

X_test_no_road.head()
test_pred = model.predict(X_test_no_road)
# Save test predictions to file

output = pd.DataFrame({'ID': test_ids,

                       'TARGET': test_pred})

output.to_csv('submission.csv', index=False)
output.head()