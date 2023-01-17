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
# import the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df=pd.read_csv("../input/canada_per_capita_income.csv")

df

#converting the x into 2D and Y into 1D

prediction=LinearRegression()

x=df.iloc[:,0:1].values

y=df.iloc[:,1].values

prediction.fit(x,y)

#assigning the m and c and plotting them 

m=prediction.coef_

c=prediction.intercept_

y_predict=m*x+c

plt.plot(x,y_predict,c="blue")

plt.scatter(x,y)
#finding the percapitalincome of canada in the year 2020

print(m*2020+c)