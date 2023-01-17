# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

import os





# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/train.csv")

#train_data.head()

test_data = pd.read_csv("../input/test.csv")

#test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



#print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



#print("% of men who survived:", rate_men)





y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('z_submission.csv', index=False)

print("Your submission was successfully saved!")