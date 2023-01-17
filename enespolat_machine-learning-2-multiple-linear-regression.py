# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/multiple-linear-regression-dataset.csv')
data.head(8)
y=data.maas.values.reshape(-1,1)
x=data.iloc[:,[0,2]].values
x
multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)
print("b0\t:",multiple_linear_regression.intercept_)
print("b1, b2\t:",multiple_linear_regression.coef_)

#deneyim ve yas değerlerini yazarak predict yapalım
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))