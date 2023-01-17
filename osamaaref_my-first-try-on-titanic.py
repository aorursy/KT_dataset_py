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
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

print(train_data.columns)

train_data.head()

#train_data.dropna(axis=0)

print(train_data.shape)



features = ["Pclass", "SibSp", "Parch"]

target_column = "Survived"



X = train_data[features]

test_X = test_data[features]

y = train_data[target_column]
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)



print(train_X.shape)

print(val_X.shape)

print(train_y.shape)

print(val_y.shape)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(random_state=1)

model.fit(train_X, train_y)
predictions = model.predict(val_X)



from sklearn.metrics import accuracy_score



print(accuracy_score(val_y, predictions))
predictions = model.predict(test_X)

print(predictions.shape)

predictions[:100]



submission = pd.DataFrame({'PassengerId': test_data["PassengerId"],

                          "Survived": predictions})



submission.to_csv("gender_submission.csv", index=False)