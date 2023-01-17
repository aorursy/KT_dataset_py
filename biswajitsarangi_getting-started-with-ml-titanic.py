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
women = train_data.loc[train_data.Sex=="female"]["Survived"]

women
survival_rate_female = sum(women) / len(women)

print(" % of women survived: ",survival_rate_female)
men = train_data.loc[train_data.Sex=="male"]["Survived"]

men
survival_rate_male = sum(men) / len(men)

print(" % of men survived: ",survival_rate_male)
y_train = train_data["Survived"]

y_train
feat = ["Pclass", "Sex", "SibSp", "Parch"]

x_train = pd.get_dummies(train_data[feat])

x_train
x_test = pd.get_dummies(test_data[feat])

x_test
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



mod = LogisticRegression()

mod.fit(x_train,y_train)



y_pred = mod.predict(x_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)



print("Your submission was successfully saved!")
