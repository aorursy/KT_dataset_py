# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

df = pd.read_csv("../input/linear-regression-dataset/Linear Regression - Sheet1.csv")

df.shape

from sklearn.linear_model import LinearRegression

x = df.iloc[:, 0:1].values  

y = df.iloc[:, 1].values

x
MachineBrain = LinearRegression()

MachineBrain.fit(x, y)
m = MachineBrain.coef_  

c = MachineBrain.intercept_
y_predict = m*3+c
y_predict = MachineBrain.predict(x)

y_predict
y
import matplotlib.pyplot as plt
plt.scatter(x,y)

plt.plot(x, y_predict, c = "red")

plt.show()