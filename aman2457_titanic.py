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
subm_df =  pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.describe()
train_df
train_df.loc[:][(train_df.Sex == 'male')  & (train_df.Survived == 1)]
survived_women = train_df.loc[train_df.Sex == "female"]["Survived"]

survived_women_rate = sum(survived_women)/len(survived_women)

print(survived_women_rate)
survived_men = train_df.loc[train_df.Sex == "male"]["Survived"]

survived_men_rate = sum(survived_men)/len(survived_men)

print(survived_men_rate)
from sklearn.ensemble import RandomForestClassifier



y = train_df["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_df[features])

X_test = pd.get_dummies(test_df[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")