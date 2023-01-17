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
data = pd.read_csv('../input/linear-regression-dataset.csv')
data
data.info
data.describe()
data.head(5)
data.tail()
data.corr()
plt.scatter(data.experience, data.salary)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
data_plot = data.loc[:,["experience","salary"]]
data_plot.plot()
data.plot(kind = "hist",y = "experience",bins = 50,range= (0,50),normed = True)
f,ax = plt.subplots(figsize=(20, 10))
sns.heatmap(data, annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
x = data.experience.values.reshape(-1,1)
y = data.salary.values.reshape(-1,1)
linear_reg.fit(x,y)
next_salary = linear_reg.predict([[20]])
print(next_salary)