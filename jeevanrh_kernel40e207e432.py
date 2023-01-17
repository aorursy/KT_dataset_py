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

train_data
train_data=train_data.drop(["Cabin"],axis=1)

train_data=train_data.dropna()

train_data.shape
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data
test_data=test_data.drop(["Cabin"],axis=1)

test_data=test_data.dropna()

test_data.shape
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data
train_data['Sex']= pd.get_dummies(train_data['Sex'])

train_data
test_data['Sex']= pd.get_dummies(train_data['Sex'])

test_data
test_data=test_data.dropna()

test_data.isna().sum()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Age"]

X=train_data[features]

X_test = test_data[features]



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")