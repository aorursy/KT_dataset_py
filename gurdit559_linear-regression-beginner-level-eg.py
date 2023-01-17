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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
df=pd.read_csv("../input/canada-per-capita-income-single-variable-data-set/canada_per_capita_income.csv")

df.tail()
df.isnull().sum()

plt.scatter(x=df['year'],y=df['per capita income (US$)'],marker="+")

plt.xlabel("Year")

plt.ylabel("per capita income")
reg=linear_model.LinearRegression()

reg.fit(df[['year']],df[["per capita income (US$)"]])

reg.predict([[2000]])
reg.coef_
reg.intercept_
plt.xlabel("Year")

plt.ylabel("per capita income")

plt.scatter(x=df['year'],y=df['per capita income (US$)'],marker="+")

plt.plot(df.year,reg.predict(df[["year"]]),color="blue")