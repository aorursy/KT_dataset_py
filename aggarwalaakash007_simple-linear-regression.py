# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
print (data)
my_data = data.dropna(how = 'any')
my_data.shape
print(my_data)
x_train = my_data.iloc[: , 0:1]
y_train = my_data.iloc[: , 1:]
print(x_train.shape)
print(y_train.shape)
test_data = np.genfromtxt("../input/test.csv" , delimiter = ',')
x_test = test_data[1: , 0:1]
y_test = test_data[1: , 1:]
print(x_test.shape)
print(y_test.shape)
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(r2_score(y_test , model.predict(x_test)))
