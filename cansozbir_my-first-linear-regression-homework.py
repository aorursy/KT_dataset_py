##Is there a relationship between the daily minimum and maximum temperature? 
##Can we predict the maximum temperature given the minimum temperature?
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1=pd.read_csv("../input/Summary of Weather.csv")
#Let's look our dataframe.
df1.head()
lr = LinearRegression()
x=df1.MinTemp.values
x.shape
#We can see our x's shape is (119040,). That's mean is (119040 , 1 )
x=x.reshape(-1,1)
x.shape
#Now in (119040,1) format.
y=df1.MaxTemp.values.reshape(-1,1)
lr.fit(x,y)
# X is min temperatures given from us.
X=np.array([10,20,30,40,50]).reshape(-1,1)
print("Results")
for i in X:
    print("Min:",i,"Predicted Max:",lr.predict([i]))
#Visualize
X
plt.scatter(x,y)
plt.show()
y_head=lr.predict(X)
plt.scatter(X,y_head,color="red")
plt.show()
