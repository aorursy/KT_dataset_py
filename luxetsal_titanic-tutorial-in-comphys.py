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
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
# Searching missing data



print("train_data missing")

missing_val_count_by_column = (train_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])



print("\n")

print("test_data missing")

missing_val_count_by_column = (test_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
train_data
# Age column

# NaN -> 평균으로 채우기



#new_train_data = train_data.copy()



mean = 29.7



train_data["Age"].fillna(value=mean, inplace=True)

print(train_data["Age"])



test_data["Age"].fillna(value=mean, inplace=True)

print(test_data["Age"])

# Fare column

# NaN -> 평균으로 채우기

# Class 1 평균 : 84.15

# Class 2 평균 : 20.66

# Class 3 평균 : 13.68



# test data에 1개 missing



test_data[test_data["Fare"].isnull()]

test_data["Fare"][152] = 13.68
y = train_data["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])
from sklearn.svm import SVC # support vector classification

import matplotlib.pyplot as plt



model=SVC(kernel='rbf',gamma='auto')

model.fit(X,y)

#plt.scatter(X["Fare"],X["Pclass"],c=model.predict(X))

plt.scatter(X["Age"],X["Fare"],c=model.predict(X))

model.score(X,y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")