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
data=pd.read_csv('/kaggle/input/petrol-consumption/petrol_consumption.csv')

data.head(2)
data.info()
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.model_selection import train_test_split





y=data.pop('Petrol_Consumption')

x=data



train_x,test_x,train_y,test_y=train_test_split(x,y)



model=RandomForestRegressor(n_estimators=10)

model.fit(train_x,train_y)

print(model.score(test_x,test_y))

model=RandomForestClassifier(n_estimators=10)

model.fit(train_x,train_y)

print(model.score(test_x,test_y))
