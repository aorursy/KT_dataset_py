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
df= pd.read_csv('../input/years-of-experience-and-salary-dataset/Salary_Data.csv')
df.head()
cdf=df[['YearsExperience','Salary']]
viz=cdf[['YearsExperience','Salary']]
viz.hist()
plt.show()
plt.scatter(cdf.YearsExperience,cdf.Salary,color="blue")
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

plt.scatter(train.YearsExperience,train.Salary)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

from sklearn import linear_model
regr= linear_model.LinearRegression()
train_x= np.asanyarray(train[['YearsExperience']])
train_y= np.asanyarray(train[['Salary']])
regr.fit(train_x,train_y)
print('Coefficient:',regr.coef_)
print('Intercept',regr.intercept_)
plt.scatter(train.YearsExperience, train.Salary, color="blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['YearsExperience']])
test_y = np.asanyarray(test[['Salary']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )