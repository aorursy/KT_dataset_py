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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
customers = pd.read_csv('../input/Ecommerce Customers')
customers.head()
customers.describe()
customers.info()
sns.set_palette("GnBu_d")

sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Length of Membership',kind="hex",data=customers)
sns.pairplot(customers)
# Length of Membership 
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(y_test,predictions))

print('MSE :'," ", metrics.mean_squared_error(y_test,predictions))

print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,predictions)))
ax1=sns.distplot(y_test,hist=None,color='r',label='Actual Values')

sns.distplot(predictions,hist=False,color='b',label='Fitted Values',ax=ax1)

#sns.distplot(y_test - predictions,bins=50)
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients