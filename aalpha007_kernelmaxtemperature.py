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
w=pd.read_csv("../input/Summary of Weather.csv")
w.info()
from sklearn.linear_model import LinearRegression



# linear regression model

linear_reg = LinearRegression()



data=w[["MinTemp","MaxTemp"]]
data
x=data.MinTemp.values.reshape(-1,1)
x
y=data.MaxTemp.values
y
y=y.reshape(-1,1)
y
linear_reg.fit(x,y)
y_predict=linear_reg.predict(x)
from sklearn.metrics import r2_score



print("r_square score: ", r2_score(y,y_predict))
linear_reg.predict([[35]])