import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
df.head()
df.describe()
df.info()
sns.jointplot(df['Time on Website'],df['Yearly Amount Spent'])
sns.jointplot(df['Time on App'],df['Yearly Amount Spent'])
sns.jointplot(df['Time on App'],df['Length of Membership'],kind='hex')
sns.pairplot(df)
sns.lmplot('Yearly Amount Spent','Length of Membership',df)
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=101)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
print('Linear Model Coefficient (m) :',regressor.coef_)
print('Linear Model Coefficient (b) :',regressor.intercept_)
predictions = regressor.predict(X_test)
plt.scatter(y_test,predictions,color='red')
plt.ylabel('Predicted values')
plt.xlabel('Y Test (True Values)')
from sklearn import metrics
metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
metrics.explained_variance_score(y_test,predictions)
sns.distplot((y_test-predictions))
data = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficients'])
data