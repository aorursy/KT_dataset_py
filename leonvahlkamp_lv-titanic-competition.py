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
# import data and show first rows

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()
# import data and show first rows

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
# check gender pattern

women = df_train.loc[df_train.Sex =='female'] ['Survived']

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = df_train.loc[df_train.Sex =='male'] ['Survived']

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = df_train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(df_train[features])

X_test = pd.get_dummies(df_test[features])



model = RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('LV_Titatnic_Test_Submission.csv', index=False)

print("Your submissionw as successfully saved!")