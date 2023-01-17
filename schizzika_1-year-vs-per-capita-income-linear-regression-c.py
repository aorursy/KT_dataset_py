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

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
df = pd.read_csv("/kaggle/input/canada-per-capita-income-single-variable-data-set/canada_per_capita_income.csv")

df.head()
plt.xlabel('Year', fontsize = 20)

plt.ylabel('Per Capita Income(US$)', fontsize = 20)

plt.scatter(df['year'], df['per capita income (US$)'], color = 'red', marker = '+')
reg = linear_model.LinearRegression()

reg.fit(df[['per capita income (US$)']], df.year)
reg.predict([[2020]])
reg.coef_
reg.intercept_
0.00107538 + (1972.6536140098344 * 2020)