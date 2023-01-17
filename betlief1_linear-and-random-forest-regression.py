# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
data.head()
data.tail()
data.info()
data['class'].unique()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()
x = data.iloc[:,[0,1,2,3,4,5]].values
y = data.iloc[:,6].values.reshape(-1,1)

y1 = [1 if i=='Abnormal' else 2 for i in y]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x,y1)
y1_ = rf.predict(x)
# If it approaches 2, the values will be normal. The closer it is to 1, the more abnormal the values.
# Result for normal values
rf.predict(np.array([[34,5,50,29,150,-0.1]]))
# Result for abnormal values
rf.predict(np.array([[63,22,39,40,98,-0.2]]))
a = data.iloc[:,2].values.reshape(-1,1)
b = data.iloc[:,3].values.reshape(-1,1)

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

linear_reg.fit(a,b)

x_ = np.arange(min(a),max(a),0.01).reshape(-1,1)
y_ = linear_reg.predict(x_)
b_ = linear_reg.predict(b)

plt.scatter(a,b)
plt.plot(x_,y_, color="red")
plt.show()
# What might be the value of lumbar lordosis angle if the sacral slope value is 56?

linear_reg.predict(np.array([56]).reshape(-1,1))
from sklearn.metrics import r2_score


#If r2_square score is close to 1, our algorithm has made so few mistakes.
#Random Forest algorithm will give more accurate results.
print("r_square score of linear regression: ", r2_score(b,b_))
print("r_square score of random forest regression: ", r2_score(y1,y1_))