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
df1=pd.read_csv("../input/Summary of Weather.csv")
from sklearn.linear_model import LinearRegression



lr = LinearRegression()
x=df1.MinTemp.values
x.shape
x=x.reshape(-1,1)
x.shape
y=df1.MaxTemp.values.reshape(-1,1)
lr.fit(x,y)
X=np.array([10,20,30,40,50]).reshape(-1,1)
print("Results")

for i in X:

    print("Min:",i,"Predicted Max:",lr.predict([i]))
#Visualize

import matplotlib.pyplot as plt

X

plt.scatter(x,y)

plt.show()
y_head=lr.predict(X)

plt.scatter(X,y_head,color="red")

plt.show()