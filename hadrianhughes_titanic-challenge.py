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
training_data = pd.read_csv("/kaggle/input/titanic/train.csv")

training_data.head()
one_sib_sp = training_data.loc[training_data.SibSp == 1]["Survived"]

two_sib_sp = training_data.loc[training_data.SibSp == 2]["Survived"]

three_sib_sp = training_data.loc[training_data.SibSp == 3]["Survived"]

four_sib_sp = training_data.loc[training_data.SibSp == 4]["Survived"]



print("% of 1 SibSp who survived: ", sum(one_sib_sp)/len(one_sib_sp))

print("% of 2 SibSp who survived: ", sum(two_sib_sp)/len(two_sib_sp))

print("% of 3 SibSp who survived: ", sum(three_sib_sp)/len(three_sib_sp))

print("% of 4 SibSp who survived: ", sum(four_sib_sp)/len(four_sib_sp))
age = 11

children = training_data.loc[training_data.Age < age]["Survived"]

adults = training_data.loc[training_data.Age >= age]["Survived"]



print("% of children who survived: ", sum(children)/len(children))

print("% of adults who survived: ", sum(adults)/len(adults))



mask = (training_data.Age.notnull())

mean_age = training_data.loc[mask, 'Age'].mean()



training_data.loc[training_data.Age.isnull(), 'Age'] = mean_age
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = training_data.loc[training_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]



t_y = training_data["Survived"]

t_X = training_data[features]



train_X, val_X, train_y, val_y = train_test_split(t_X, t_y, random_state=1)



train_X_D = pd.get_dummies(train_X)

val_X_D = pd.get_dummies(val_X)
def mae_with_depth(depth):

    mae_model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=1)

    mae_model.fit(train_X_D, train_y)

    return mean_absolute_error(mae_model.predict(val_X_D), val_y)



print("Max depth of 5, MAE: ", mae_with_depth(5))

print("Max depth of 3, MAE: ", mae_with_depth(3))

print("Max depth of 1, MAE: ", mae_with_depth(1))



model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)

model.fit(train_X_D, train_y)
mask = (test_data.Age.notnull())

mean_age = test_data.loc[mask, 'Age'].mean()



test_data.loc[test_data.Age.isnull(), 'Age'] = mean_age
test_X = pd.get_dummies(test_data[features])



predictions = model.predict(test_X)



output = pd.DataFrame({ 'PassengerId': test_data.PassengerId, 'Survived': predictions })

output.head()
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")