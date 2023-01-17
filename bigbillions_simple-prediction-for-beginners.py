# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#loading our dataset
data = "../input/salary.csv"
df = pd.read_csv(data)
df.head()
#Now we want to visualize our dataset 
import numpy as np
import matplotlib.pyplot as plt


xi = (df['year'])
yi= (df['salary'])

x = xi.values.reshape(-1, 1)
y = yi.values.reshape(-1, 1)

plt.figure()
plt.title('Salary and year dataset')
plt.xlabel('Salary')
plt.ylabel('Year')
plt.plot(x, y, 'k.')
plt.grid(True)
plt.show()
#Now we will train our model and make prediction
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)
print(model.predict(50))