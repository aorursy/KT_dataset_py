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
dataset=pd.read_csv('../input/train.csv')
dataset1=pd.read_csv('../input/test.csv')
x1=dataset1.iloc[:,[1,3]].values
x=dataset.iloc[:,[2,4]].values
y=dataset.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
#////////////////////
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
x1[:,1]=labelencoder_x1.fit_transform(x1[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x1=onehotencoder.fit_transform(x1).toarray()
#////////////////
from sklearn.linear_model import LinearRegression
raja=LinearRegression()
raja.fit(x,y)
y_pred=raja.predict(x1)



x1