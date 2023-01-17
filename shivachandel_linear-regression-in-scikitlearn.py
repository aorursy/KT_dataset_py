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
data =pd.read_csv('../input/USA_Housing.csv')
data.describe()
feature=data.drop(['Price','Address'],axis=1).values
target =data['Price'].values
from sklearn.model_selection import train_test_split
train,test,train_label,test_label=train_test_split(feature,target,test_size=0.33,random_state=222)
from sklearn.linear_model import LinearRegression
reg=LinearRegression(fit_intercept=True)
model = reg.fit(train,train_label)
predict = model.predict(test)
from sklearn.metrics import r2_score
print(r2_score(test_label,predict))
