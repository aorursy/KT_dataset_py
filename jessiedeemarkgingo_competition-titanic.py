# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

train_data = pd.read_csv("../input/titanic/train.csv")
test_data.head()
train_data.head()
gender_submission.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)    
from xgboost import XGBClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,

       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,

       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,

       objective='binary:logistic', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=1, seed=0, silent=True, subsample=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':predictions})

output.to_csv("my_submissionv2.csv", index=False)



print("Your submission was successfully saved!")
