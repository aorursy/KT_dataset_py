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
data=pd.read_csv("../input/pyramid_scheme.csv",sep=",")

display(data)

data.drop(data.columns[0],axis=1,inplace=True)
display(data)
data.drop(data.columns[0],axis=1,inplace=True)
display(data)
x=data.iloc[:,[0,2]].values
y=data.profit.values
from sklearn.linear_model import LinearRegression

Linear_reg=LinearRegression()
Linear_reg.fit(x,y)

y_head=Linear_reg.predict(x)
from sklearn.metrics import r2_score

print("r square score:",r2_score(y,y_head))