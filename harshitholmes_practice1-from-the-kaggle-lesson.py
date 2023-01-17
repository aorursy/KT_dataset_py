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
dt = pd.read_csv('../input/train.csv')
dt = dt.dropna()
print(dt)
y = dt.PassengerId
dtf = ['Survived', 'Pclass', 'Age','SibSp', 'Fare']
x = dt[dtf]
x.describe()
x.head()

from sklearn.tree import DecisionTreeRegressor as dtr

dtm = dtr(random_state = 1)
dtm.fit(x, y)



from sklearn.metrics import mean_absolute_error as mae

pdp = dtm.predict(x)
mae(y, pdp)
from sklearn.model_selection import train_test_split as tts 

train_x, val_x, train_y, val_y = tts(x, y, random_state = 0)
mdl = dtr()

mdl.fit(train_x, train_y)

vpd = mdl.predict(val_x)
print(mae(val_y, vpd))
