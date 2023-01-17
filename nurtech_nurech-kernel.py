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
print ("Hello")
import numpy as np



x=np.array([2,4,3,5,6])

y=np.array([10,5,9,4,3])



E_x=np.mean(x)

E_y=np.mean(y)



cov_xy=np.mean(x*y)- E_x*E_y



y_0= E_y- cov_xy / np.var(x)* E_x

m= cov_xy/np.var(x)



y_pred=m*x+y_0



print("E[(y_pred-y_actual)^2]=", np.mean(np.square(y_pred-y)))
import numpy as np

import matplotlib.pyplot as plt



# Create data

N = 500

x = np.random.rand(N)

y = np.random.rand(N)

colors = (0,0,0)

area = np.pi*3



# Plot

plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.title('Scatter plot pythonspot.com')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
import numpy as np

import matplotlib.pyplot as plt



x=np.array([2,4,3,5,6])

y=np.array([10,5,9,4,3])



E_x=np.mean(x)

E_y=np.mean(y)



cov_xy=np.mean(x*y)- E_x*E_y



y_0= E_y- cov_xy / np.var(x)* E_x

m= cov_xy/np.var(x)



y_pred=m*x+y_0





N = 500

colors = (0,0,0)

area = np.pi*3



# Plot

plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.title('Homework: linear regression prediction')

plt.xlabel('x')

plt.ylabel('y')

plt.plot(x, y_pred, color='Green', alpha=0.5)

plt.show()