import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import numpy as np

import pandas as pd

customers = pd.read_csv("../input/dataset/FyntraCustomerData.csv")
customers.head()
customers.describe()
customers.info()
#Check Correlations

# More time on site, more money spent.

sns.jointplot(x='Time_on_Website',y='Yearly_Amount_Spent',data=customers)
correlation = customers.corr()
sns.heatmap(correlation, cmap="YlGnBu")
sns.jointplot(x='Time_on_App',y='Yearly_Amount_Spent',data=customers)

# This one looks stronger correlation than Time_on_Website
sns.pairplot(customers)
sns.lmplot(x='Length_of_Membership',y='Yearly_Amount_Spent',data=customers)
sns.jointplot(x='Length_of_Membership', y='Yearly_Amount_Spent', data=customers,kind="kde")
X = customers[['Avg_Session_Length', 'Time_on_App','Time_on_Website', 'Length_of_Membership']]
y = customers['Yearly_Amount_Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=85)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
#calculating the residuals

print('y-intercept             :' , lm.intercept_)

print('beta coefficients       :' , lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test ')

plt.ylabel('Y Predicted ')
# here You can check the values Test vrs Prediction

dft = pd.DataFrame({'Y test': y_test, 'Y Pred':predictions})

dft.head(10)
# calculate these metrics by hand!

from sklearn import metrics



print('Mean Abs Error MAE      :' ,metrics.mean_absolute_error(y_test,predictions))

print('Mean Sqrt Error MSE     :' ,metrics.mean_squared_error(y_test,predictions))

print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,predictions)))

print('r2 value                :' ,metrics.r2_score(y_test,predictions))
sns.distplot((y_test-predictions),bins=50);
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients