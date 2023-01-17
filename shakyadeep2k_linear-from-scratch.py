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
import matplotlib.pyplot as plt

import numpy as np



x,y = [1,2,4,3,5],[1,3,3,2,5]
plt.scatter(x,y)

plt.show()
x_mean = np.mean(x)

y_mean = np.mean(y)



x = pd.Series(x)

y = pd.Series(y)



m = sum((x - x_mean) * (y - y_mean))/sum((x - x_mean)**2)

c = y_mean - x_mean*m



y_pred = x*m + c



plt.scatter(x,y)

plt.plot(x,y_pred)

plt.show()



r2 = sum((y_pred - y_mean)**2)/sum((y-y_mean)**2)

mse = sum((y_pred-y)**2)/x.size

rmse = mse**0.5



print(r2)

print(mse)

print(rmse)