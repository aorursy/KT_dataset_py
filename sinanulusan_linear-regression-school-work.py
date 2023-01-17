# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import data

data = pd.read_csv("/kaggle/input/sinan-dataset/multiple_linear_regression_dataset.csv",sep = ";")

data 
data.info()
#plot data

plt.scatter(data.deneyim,data.maas)

plt.xlabel("deneyim")

plt.ylabel("maas")

plt.show()
# linear regression



#sklearn library

from sklearn.linear_model import LinearRegression



#linear regression model

linear_reg = LinearRegression()



x = data.deneyim.values.reshape(-1,1)

y = data.maas.values.reshape(-1,1)



linear_reg.fit(x,y)
# prediction

import numpy as np



a = linear_reg.predict([[0]])

print("a: ",a)



a_ = linear_reg.intercept_

print("a_: ",a_)   #y eksenini kestiği nokta ing:intercept

b = linear_reg.coef_

print("b: ",b)  #egim ing:slope 
#maas = 1663+1138*deneyim



maas_yeni = 1663+1138*11

print(maas_yeni)



print(linear_reg.predict([[11]]))
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #x axis(deneyim)

plt.scatter(x,y)





y_head = linear_reg.predict(array) #y axis (maas)



plt.plot(array, y_head,color = "red" )



plt.show()
from sklearn.linear_model import LinearRegression





x = data.iloc[:,[0,2]].values

y = data.maas.values.reshape(-1,1)

# fitting data

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)



print("a: ", multiple_linear_regression.intercept_)

print("b1,b2: ",multiple_linear_regression.coef_)
# predict

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))
