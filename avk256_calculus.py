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
A = [[1,2],[3,4]]

B = [[5,6],[7,8]]



mA = np.matrix(A)

mB = np.matrix(B)



mC = np.matmul(mA, mB)

print(mC)
from sklearn.metrics import mean_squared_error



y_true = [12,32,5,67,3,6]

y_pred = [10,31,7,69,2,8]



mse_1 = mean_squared_error(y_true, y_pred)

print('mse1 = ',mse_1)



mse_2 = sum(list(map(lambda x, y: (x-y)**2, y_true, y_pred)))/len(y_true)

print('mse2 = ', mse_2)
from scipy import stats

array = [10, 12]



average_1 = np.mean(array)

std_1 = np.std(array)



print("Avarage value_1 = ", average_1)

print("Standard deviation_1 = ", std_1)



average_2 = sum(array)/len(array)

std_2 = (sum(list(map(lambda x: (x-average_2)**2, array)))/len(array))**(1/2)



print("Avarage value_2 = ", average_2)

print("Standard deviation_2 = ", std_2)





import matplotlib.pyplot as plt 

import seaborn as sns

advertising = pd.read_csv("../input/advertising-dataset/advertising.csv")



advertising.head()



with open('../input/advertising-dataset/advertising.csv') as f:

    alist = [line.rstrip() for line in f]
print(alist)
type(alist)
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')

plt.show()
x_list = list(advertising['TV'])

y_list = list(advertising['Sales'])

print(x_list)

print(y_list)

x_mean = sum(x_list)/len(x_list)

y_mean = sum(y_list)/len(y_list)

beta_1 = sum(list(map(lambda x,y: (x-x_mean)*(y-y_mean), x_list,y_list))) / sum(list(map(lambda x: (x-x_mean)**2, x_list))) 

beta_0 = y_mean - beta_1*x_mean



print('beta_1', beta_1)

print('beta_0', beta_0)
from sklearn.linear_model import LinearRegression



x_toy = [1,2,3,4,5]

y_toy = [2,4,6,8,10]



lr = LinearRegression()

x_list1 = np.transpose(np.atleast_2d(x_toy))

lr.fit(x_list1,y_toy)

print('beta_1',lr.coef_)

print('beta_0',lr.intercept_)
