#leave a comment help me to get better , thank you 
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
train_data =pd.read_csv("/kaggle/input/random-linear-regression/train.csv")

test_data =pd.read_csv("/kaggle/input/random-linear-regression/test.csv")
train_data.isnull().sum()

x_train = train_data['x']

y_train =train_data ['y']

x_test =test_data['x']

y_test =test_data['y']
print('the mean of y_train is :',y_train.mean(),'\n')

print('the median of y_train is :',y_train.median(),'\n')

print("=========================================================================================")

print('the mean of x_train is :',x_train.mean(),'\n')

print('the median of x_train is :',x_train.median())

print("=========================================================================================")

print('the mean of x_test is :',x_test.mean(),'\n')

print('the median of x_test is :',x_test.median())

print("=========================================================================================")

print('the mean of y_test is :',y_test.mean(),'\n')

print('the median of y_test is :',y_test.median())

print("=========================================================================================")


y_train= y_train.fillna(y_train.mean())

y_trian.isnull().sum()

x_train.isnull().sum()
import matplotlib.pyplot as plt

plt.title("trainning data set",fontsize =15)

plt.scatter(x_train,y_train ,color ='red')

plt.show()





from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

model = LinearRegression()

model.fit(np.array(x_train).reshape(-1,1) ,np.array(y_train).reshape(-1,1))

y_predicted =model.predict(np.array(x_test).reshape(-1,1))

r2_score(y_test,y_predicted)


