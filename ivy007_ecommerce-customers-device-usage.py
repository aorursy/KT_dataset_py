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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
customers = pd.read_csv('../input/Ecommerce Customers')
customers.head()
customers.info()
sns.jointplot(data = customers, x = 'Time on App', y = 'Yearly Amount Spent')
sns.jointplot(data = customers, x = 'Time on App', y = 'Yearly Amount Spent')
sns.jointplot(x='Time on App', y='Length of Membership', kind = 'hex', data = customers)







sns.pairplot(customers)
sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers);
customers.columns
X = customers[[ 'Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)
Predictions = lm.predict(X_test)
plt.scatter(y_test,Predictions,)
from sklearn import metrics
print('MAE :' ,metrics.mean_absolute_error(y_test,Predictions))
print('MSE :' ,metrics.mean_squared_error(y_test,Predictions))
print('MAE :' ,np.sqrt(metrics.mean_squared_error(y_test,Predictions)))
sns.distplot(y_test - Predictions, bins = 40)
sd_plt = pd.DataFrame(lm.coef_,X.columns,columns =['Coefficients'])
sd_plt