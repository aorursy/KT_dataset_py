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
%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
df=pd.read_csv("../input/car-price-prediction3-features/carprices.csv")

df
el=LabelEncoder()

dlf=df

dlf["Car Model"]=el.fit_transform(dlf["Car Model"])

dlf
x=np.array(dlf[["Car Model","Mileage","Age(yrs)"]])

x
y=np.array(dlf["Sell Price($)"])

y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

x_test
transformer=ColumnTransformer( transformers=[('OneHot',OneHotEncoder(),[0])],remainder="passthrough")

X=transformer.fit_transform(x_train)

x_test=transformer.fit_transform(x_test)

X
X=X[:,1:]

X
reg=LinearRegression()

reg.fit(X,y_train)
reg.predict([[0,0,70000,6]])
reg.score(X,y_train)
x_test
x_test=x_test[:,1:]

x_test
y_test
reg.fit(x_test,y_test)
reg.predict([[0,0,70000,6]])
reg.score(x_test,y_test)