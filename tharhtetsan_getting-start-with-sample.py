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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head()
test_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df.head()


train_women_survived = train_df.loc[train_df.Sex == 'female']["Survived"]

train_men_survived = train_df.loc[train_df.Sex == 'male']["Survived"]



tr_men_survival_rate = sum(train_men_survived)/len(train_men_survived) * 100

tr_women_survival_rate = sum(train_women_survived)/len(train_women_survived) * 100



print("Men survived rate in training set: {:.2f}%".format(tr_men_survival_rate))

print("Women survived rate in training set: {:.2f}%".format(tr_women_survival_rate))





test_women_survived = test_df.loc[test_df.Sex == 'female']["Survived"]

test_men_survived = test_df.loc[test_df.Sex == 'male']["Survived"]



te_men_survival_rate = sum(test_men_survived)/len(test_men_survived) * 100

te_women_survival_rate = sum(test_women_survived)/len(test_women_survived) * 100



print("Men survived rate in test set: {:.2f}%".format(te_men_survival_rate))

print("Women survived rate in test set: {:.2f}%".format(te_women_survival_rate))
from sklearn.ensemble import RandomForestClassifier



y = test_df["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(test_df[features])

X_test = pd.get_dummies(test_df[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")