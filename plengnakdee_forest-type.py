import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
train.head()
train.describe()
for column in train.columns:

    print(column, train[column].nunique())
sns.pairplot(train[['Cover_Type', 'Elevation', 'Slope', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']])
sns.pairplot(train[['Cover_Type', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']])
sns.pairplot(train[['Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']])
#train = train.drop(["Id"], axis = 1)



#test_ids = test["Id"]

#test = test.drop(["Id"], axis = 1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_val, y_train, y_val = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.2)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
model = RandomForestClassifier(n_estimators=150)

model.fit(X_train, y_train)
from sklearn.metrics import classification_report, accuracy_score
model.score(X_train, y_train)
predictions = model.predict(X_val)

accuracy_score(y_val, predictions)
test.head()
test_pred = model.predict(test)
print(test_pred)
# Save test predictions to file

output = pd.DataFrame({'ID': test_ids,

                       'TARGET': test_pred})

output.to_csv('submission.csv', index=False)