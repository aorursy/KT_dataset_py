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

import pandas as pd

test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


x=train.drop(['Name','Cabin','Ticket','PassengerId','Survived'], axis=1)
y=train.Survived
test1 =test.drop(['Name','Cabin','Ticket','PassengerId'], axis=1)
x1 = pd.get_dummies(x)
test2= pd.get_dummies(test1)


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(x1)
imp.fit(test2)
x2 = imp.transform(x1)
test3= imp.transform(test2)


from sklearn.ensemble import GradientBoostingClassifier
model_rf = GradientBoostingClassifier(random_state=1211)
model_rf.fit( x2 , y )
y_pred = model_rf.predict(test3)


id1= test.PassengerId
Label = y_pred
Label = pd.Series(Label)

submit = pd.concat([id1,Label],axis=1, ignore_index=True)
submit.columns=['PassengerId','Survived']

submit.to_csv("submit.csv",index=False)
