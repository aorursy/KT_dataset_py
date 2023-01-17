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
x=[1,2,3,4,5]

y=[1,3,3,2,5]
import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.show()
x= pd.Series(x)

y= pd.Series(y)

x_mean = np.mean(x)

y_mean = np.mean(y)

m = sum((x-x_mean)*(y-y_mean))/sum(((x-x_mean)**2))

c = y_mean - m*x_mean

y_predict = x*m +c

print("c :",c)

print("m" ,m)

plt.scatter(x,y)

plt.plot(x,y_predict)

plt.show()

mean_square_error = sum((y_predict-y)**2)/x.size

root_mean_square_error = (mean_square_error)**0.5

r_2= sum((y_predict-y_mean)**2) / sum((y-y_mean)**2)

print("MSE :",mean_square_error)

print("RMSE :", root_mean_square_error)

print("r2 :" ,r_2)