# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting data
import seaborn as sns # Advanced visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/panait/panait.csv')
data
data.info
data.describe()
data.head(5)
data.tail()
data.corr()
plt.scatter(data.date, data.rate)
plt.xlabel("date")
plt.ylabel("rate")
plt.show()
data_plot = data.loc[:,["date","rate"]]
data_plot.plot()
data.plot(kind = "hist",y = "rate",bins = 50,range= (0,0.0078),normed = True)
f,ax = plt.subplots(figsize=(20, 10))
sns.heatmap(data, annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
x = data.date.values.reshape(-1,1)
y = data.rate.values.reshape(-1,1)
linear_reg.fit(x,y)
rate = linear_reg.predict([[20]])
print(rate)