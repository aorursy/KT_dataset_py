# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/ecommerce-customers/Ecommerce Customers.csv")
df.head(5)
df.info()
df.describe()
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)
sns.jointplot('Time on App','Yearly Amount Spent',data=df)
sns.jointplot('Time on App','Length of Membership',data=df,kind='hex')
sns.pairplot(df)
sns.lmplot('Length of Membership','Yearly Amount Spent',data=df)
df.columns
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
predictions = lm.predict(X_test)
sns.scatterplot(y_test,predictions)
MAE = metrics.mean_absolute_error(y_test,predictions)
MSE = metrics.mean_squared_error(y_test,predictions)
RSME = np.sqrt(metrics.mean_squared_error(y_test,predictions))
print('MAE: ' + str(MAE))
print('MSE: ' + str(MSE))
print('RSME: ' + str(RSME))

metrics.explained_variance_score(y_test,predictions)
sns.distplot((y_test-predictions),bins=50)
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

