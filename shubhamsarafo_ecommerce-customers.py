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
df = pd.read_csv('../input/Ecommerce Customers')
df.head()
df.columns
df.describe()
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.jointplot(y='Yearly Amount Spent',x='Time on Website',data=df)
sns.jointplot(y='Yearly Amount Spent',x='Time on App',data=df)
df.columns
sns.jointplot(y='Length of Membership',x='Time on App',data=df,kind='hex')
sns.pairplot(df)
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
y = df['Yearly Amount Spent']

X = df[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm =LinearRegression()
#fit the training data to the model

lm.fit(X_train,y_train)
lm.coef_
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test (TRUE VALUES)')

plt.ylabel('PREDICTED VALUES')
#EVALUATE THE MODEL USING MATHEMATICS

from sklearn import metrics

print('MAE ',metrics.mean_absolute_error(y_test,predictions))

print('MSE ',metrics.mean_squared_error(y_test,predictions))

print('RMSE ',np.sqrt(metrics.mean_squared_error(y_test,predictions))) 
metrics.explained_variance_score(y_test,predictions)
#residuals

sns.distplot((y_test - predictions),bins=50)
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])

cdf
