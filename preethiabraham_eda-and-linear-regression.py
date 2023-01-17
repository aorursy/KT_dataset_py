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
import pandas as pd
filename = "/kaggle/input/ecommerce-customers/Ecommerce Customers.csv"
customers = pd.read_csv(filename)
customers.head()
customers.describe()
customers.info()
import seaborn as sns
import scipy.stats as stats
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data= customers).annotate(stats.pearsonr)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data= customers).annotate(stats.pearsonr)
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex').annotate(stats.pearsonr)
sns.pairplot(customers)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data= customers)
from sklearn.linear_model import LinearRegression
X=customers[['Avg. Session Length', 'Time on App','Time on Website','Length of Membership']]
Y=customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
lm=LinearRegression()
lm.fit(X_train, Y_train)
print('Coefficients: ', lm.coef_)
predictions=lm.predict(X_test)
predictions
sns.scatterplot(x=Y_test, y=predictions)
from sklearn import metrics
MAE=metrics.mean_absolute_error(Y_test, predictions)
MAE
MSE=metrics.mean_squared_error(Y_test, predictions)
print(MSE)
rmse=np.sqrt(MSE)
rmse
sns.distplot((Y_test-predictions), bins=40)
pd.DataFrame(lm.coef_, index=['Avg Session Length', 'Time on App', 'Time on Website','Length of Membership'], columns=['Coeffecient'])
