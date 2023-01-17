import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


customers = pd.read_csv("../input/ecommerce-customers/Ecommerce Customers.csv")
customers.head()


customers.describe()
customers.info()
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# Mere tid på hjemmesiden er lig med = flere penge brugt.
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)

sns.pairplot(customers)
sns.heatmap(customers.corr(), cmap="Blues", annot=True)

sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = customers)

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
## Implementere en instans af Linear Regression modellen og kald den LM.
lm = LinearRegression()
lm.fit(X_train,y_train)
## Printe koefficienter ud af modellen
# The coefficients
print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=50);
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients























