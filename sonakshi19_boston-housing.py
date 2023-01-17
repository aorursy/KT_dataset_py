import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
boston = pd.read_csv('../input/boston.csv')
boston.head()
boston.keys()
boston.info()
boston.describe() # For all numerical columns
sns.set_palette("GnBu_d")

sns.set_style('whitegrid')

sns.pairplot(boston,size = 2,vars = ['CRIM','NOX','RAD','DIS','MEDV'])
#Let's check the distribution plot of MEDV value 

sns.distplot(boston['MEDV'],bins = 20)
ax = plt.subplots(figsize = (14,6))

sns.heatmap(boston.corr(),cmap = 'magma',linecolor = 'white',lw = 1)
boston.corr()
sns.jointplot(x='RAD',y='TAX',data=boston,kind='scatter')
sns.jointplot(x='RAD',y='TRACT',data=boston,kind='scatter')
sns.jointplot(x='TAX',y='TRACT',data=boston,kind='scatter')
#Training the Linear Model

boston.columns
X = boston[['CRIM', 'ZN', 'INDUS',

       'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO']]

y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
# print the intercept

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
Predictions = lm.predict(X_test)

# Let's check through a scatter plot how they are aligned

plt.scatter(y_test,Predictions)
#residual Histogram

sns.distplot((y_test-Predictions),bins=50)
#Regression Evaluation Metrics

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, Predictions))

print('MSE:', metrics.mean_squared_error(y_test, Predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Predictions)))