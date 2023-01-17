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

print("hello")
import numpy as np



x = np.array([2, 4, 3, 5, 6])

y = np.array([10, 5, 9, 4, 3])



x

x +y
E_x = np.mean(x)

E_y = np.mean(y)



cov_xy = np.mean(x*y)-E_x*E_y



y_0 = E_y - cov_xy/np.var(x)*E_x

m = cov_xy/np.var(x)



y_pred=m*x+y_0



print ("E[(y_pred-y_actual)^2] = ", np.mean(np.square(y_pred-y)))

print (y_pred)
import matplotlib.pyplot as plt



# Scatter Plot

plt.scatter(x, y)

plt.title('Scatter plot')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
# Line Plot

plt.plot(x,y)

plt.title('Line Plot')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
plt.plot(x, y_pred, c="red") # best fit line

plt.scatter(x, y, c="black")

plt.title('Scatter plot with best fit line')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
plt.scatter(x, y)

plt.title('Scatter plot with best fit line v.2')

plt.xlabel('x')

plt.ylabel('y')

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x))) # best fit line diffrent code

plt.show()