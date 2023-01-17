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

import seaborn as sns

%matplotlib inline

Ecommerce_Customers = pd.read_csv("../input/Ecommerce Customers.csv")
Ecommerce_Customers.head()
Ecommerce_Customers.describe()

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=Ecommerce_Customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=Ecommerce_Customers)
sns.jointplot(x='Time on App',y='Length of Membership',data=Ecommerce_Customers,kind='hex')
sns.pairplot(Ecommerce_Customers)
sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=Ecommerce_Customers)
Ecommerce_Customers.info()
Ecommerce_Customers.dropna(inplace=True)

y=Ecommerce_Customers['Yearly Amount Spent']

X=Ecommerce_Customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

from  sklearn.model_selection import train_test_split

from  sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

lm=LinearRegression()

y_test

#X_train.shape
lm.fit(X_train,y_train)
vg=pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])

vg
predict=lm.predict(X_test)

predict
y_test

sns.scatterplot(y_test,predict)
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,predict))
print('MSE',metrics.mean_squared_error(y_test,predict))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predict)))
metrics.explained_variance_score(y_test,predict)
sns.distplot(y_test-predict,bins=50)