# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn    as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ecommerce-customers/Ecommerce Customers.csv')
df.head()
df.describe()
df.info()
sns.jointplot(data=df ,x='Time on Website' ,y='Yearly Amount Spent')
sns.jointplot(data=df ,x='Time on App' ,y='Yearly Amount Spent')
sns.jointplot(data=df ,x='Time on App' ,y='Length of Membership' ,kind='hex')
sns.pairplot(df)
df.corr()
sns.heatmap(df.corr() ,annot=True)
sns.lmplot(x='Length of Membership' ,y='Yearly Amount Spent' ,data=df)
df.columns
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership', ]]

y = df[['Yearly Amount Spent']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test(True Values)')
plt.ylabel('Predictions Values')
from sklearn import metrics
print('MAE : ' , metrics.mean_absolute_error(y_test,predictions))
print('MSE : ' , metrics.mean_squared_error(y_test,predictions))
print('RMSE : ' ,np.sqrt(metrics.mean_squared_error(y_test,predictions)))
metrics.explained_variance_score(y_test,predictions)
sns.distplot((y_test-predictions) ,bins=50)
cdf =  pd.DataFrame(lm.coef_.reshape(4,1) , X.columns , columns=['Coeffecient']) 
cdf.head()
