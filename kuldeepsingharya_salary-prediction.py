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
data=pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
data.info()
data.head()
data.shape
data.describe()
data.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.regplot(x="YearsExperience",y="Salary",data=data)
plt.figure(figsize=(16, 8))

plt.scatter(

    data['YearsExperience'],

    data['Salary'],

)

plt.xlabel("Years")

plt.ylabel("Salary")

plt.show()
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
X = data['YearsExperience'].values.reshape(-1,1)

y = data['Salary'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X, y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)

plt.figure(figsize=(16, 8))

plt.scatter(

    data['YearsExperience'],

    data['Salary'],

    c='black'

)

plt.plot(

    data['YearsExperience'],

    predictions,

    c='blue',

    linewidth=2

)

plt.xlabel("Years")

plt.ylabel("Salary ($)")

plt.show()
X = data['YearsExperience']

y = data['Salary']

X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())