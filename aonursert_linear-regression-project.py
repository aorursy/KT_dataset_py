import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
customers = pd.read_csv("../input/ecommerce-customers/Ecommerce Customers.csv")
customers.head()
customers.describe()
customers.info()
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)
sns.jointplot(x="Time on App", y="Length of Membership", data=customers, kind="hex")
sns.pairplot(customers)
# Length of Membership
sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers)
customers.head()
customers.columns
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
predictions = lm.predict(X_test)
sns.scatterplot(y_test, predictions)
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
metrics.explained_variance_score(y_test, predictions)
sns.distplot(y_test-predictions, bins=30)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coeff"])
cdf