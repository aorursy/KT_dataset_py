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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/101-simple-linear-regressioncsv/1.01. Simple linear regression.csv')
data.head()
# feature
x = data['SAT']

# target
y = data['GPA']
x_matrix = x.values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(x_matrix, y)
# R-squared
display(reg.score(x_matrix, y))

# coefficiants
display(reg.coef_)

# intercept
display(reg.intercept_)
new_data = pd.DataFrame(data = [1730, 1750], columns = ['SAT'])
reg.predict(new_data)
new_data['Predicated_GPA'] = reg.predict(new_data)
new_data
plt.scatter(x, y)
yhat = reg.coef_ * x_matrix + reg.intercept_

fig = plt.plot(x, yhat, lw = 4, c = 'orange', label = 'Regression Line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
data = pd.read_csv('../input/102-multiple-linear-regression/1.02 Multiple linear regression.csv')
data.head()
data.describe()
x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']
reg.fit(x, y)

# R-squared
r2 = reg.score(x, y)
display(reg.score(x, y))

# coefficiants
display(reg.coef_)

# intercept
display(reg.intercept_)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
adjusted_r2
from sklearn.feature_selection import f_regression
p_values = f_regression(x, y)[1]
p_values.round(3)
reg_summary = pd.DataFrame(data = x.columns.values, columns = ['Features'])
reg_summary['Coefficiants'] = reg.coef_
reg_summary['P-values'] = p_values.round(3)

reg_summary
plt.scatter(x['SAT'], y)
yhat = reg.coef_ * x + reg.intercept_

fig = plt.plot(x, yhat, lw = 4, c = 'orange', label = 'Regression Line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.ylim(2.25,4)
plt.xlim(1600, 2100)
plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled, y)
reg.coef_, reg.intercept_
reg_summary = pd.DataFrame([['Bias'], ['SAT'], ['Rand 1,2,3']], columns = ['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]

reg_summary
new_data = pd.DataFrame([[1700, 2], [1750, 3]], columns = ['SAT', 'Rand 1,2,3'])
new_data
new_scaled_data = scaler.transform(new_data)
reg.predict(new_scaled_data)
reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:,0].reshape(-1, 1)
reg_simple.fit(x_simple_matrix, y)
reg_simple.predict(new_scaled_data[:,0].reshape(-1, 1))