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

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
df.head(5)
x = df.iloc[:,0:-1]

y = df.iloc[:,[1]]
x.shape
plt.scatter(x,y,label='simple linear regression',color='r')

plt.xlabel('Years of Exp.')

plt.ylabel('salary')
reg = linear_model.LinearRegression()

score = []
for i in range(0,101):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)

    reg.fit(X_train,y_train)

    y_pred = reg.predict(X_test)

    score.append(r2_score(y_test,y_pred))
plt.plot(range(0,101),score , color="blue",linewidth=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47)

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

print(r2_score(y_test,y_pred))
plt.scatter(X_test,y_test, color = 'red' )

plt.plot(X_test,y_pred,color='black',linewidth=3)