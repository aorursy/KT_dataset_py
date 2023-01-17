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
data=np.loadtxt('../input/data.txt')
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X=data[:,:16]
y1=data[:,16]
y2=data[:,17]
X_train1,X_test1,y_train1,y_test1=train_test_split(X,y1,test_size=0.3,random_state=42)
X_train2,X_test2,y_train2,y_test2=train_test_split(X,y2,test_size=0.3,random_state=42)
r_squared1=[]
r_squared2=[]
y_pred1=[]
y_pred2=[]
models_list=[LinearRegression(),Ridge(alpha=1.0)]
for model in models_list:
    pipeline=Pipeline([('scaler',MinMaxScaler()),('lr',model)])
    pipeline.fit(X_train1,y_train1)
    r_squared1.append(pipeline.score(X_test1,y_test1))
    y_pred1.append(pipeline.predict(X_test1))
    pipeline.fit(X_train2,y_train2)
    r_squared2.append(pipeline.score(X_test2,y_test2))
    y_pred2.append(pipeline.predict(X_test2))
print(r_squared1,r_squared2)