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
import pandas as pd
data=pd.read_csv('../input/positon-salaries1/Position_Salaries.csv')

data.head()
#apply polinomial regression to above data
#step 1: visualize our data points
import matplotlib.pyplot as plt
plt.scatter(data.Level,data.Salary)

plt.xlabel("levels")

plt.ylabel("salary")

plt.title("levels vs salary")

plt.show()
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
x=data['Level']
y=data["Salary"]

y.head()
import numpy as np
x=np.array(x)

y=np.array(y)
x_poly=poly.fit_transform(x[:,np.newaxis])
x_poly
plt.scatter(data.Level,data.Salary)

plt.plot(data.Level,model.predict(x_poly))

plt.show()
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_poly,y)
x=np.array([6.5])
x.ndim
x
x1=poly.fit_transform(x[:,np.newaxis])
model.predict(x1)