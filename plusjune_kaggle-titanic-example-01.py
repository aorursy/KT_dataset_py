# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
! ls -l ../input
import pandas as pd

from sklearn.linear_model import LogisticRegression



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



df_train['Sex'] = df_train['Sex'].map({'male': 1, 'female':0})

df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female':0})



train_x = df_train[['Pclass', 'Sex']]

train_y = df_train['Survived'].values



model = LogisticRegression(solver='lbfgs')

model.fit(train_x, train_y)



y_pred = model.predict(df_test[['Pclass', 'Sex']])



submission = pd.DataFrame({

  "PassengerId": df_test["PassengerId"],

  "Survived": y_pred

})



submission.to_csv('submission.csv', index=False)
ll
!head -10 submission.csv