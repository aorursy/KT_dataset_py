# Import relevant libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
# Loading data

df = pd.read_csv('../input/Ecommerce Customers.csv')
# Data contains various information on customers of a store including time spent on the company's website, mobile app, and length of membership.

# Goal is to determine which of these features have the greatest impact on predicting the yearly amount spent.

# A linear regression model was used for this analysis.

df.head()
df.info()
df.describe()
# Check for missing values

df.isnull().sum()
# Exploratory Data Analysis

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)

plt.show()
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)

plt.show()
# Time on App has a stronger correlation to yearly amount spent vs. time on website
sns.jointplot(x='Time on App', y='Length of Membership', data=df, kind='hex')

plt.show()
sns.pairplot(df)

plt.show()
# Length of membership appears to have the strongest correlation to yearly amount spent from this visualization
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)

plt.show()
# Only columns containing numerical data will be used for this analysis

df.columns
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

X.head()
y = df['Yearly Amount Spent']

y.head()
# Generating train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Importing the model and fitting it to training data

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)
# Predicting test data

y_pred = lm.predict(X_test)
# Visualizing test values vs. predicted values

plt.scatter(y_test, y_pred)

plt.xlabel('y_test')

plt.ylabel('y_pred')

plt.show()
# Predictions look to match well with the actual results form the test data
# Model Evaluation

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))

print('MSE: ', metrics.mean_squared_error(y_test, y_pred))

print('SMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Model appears to be good and ready for deployment

# k-fold cross-validation can be another metric to evaluate model performance
# Plot of residuals

sns.distplot(y_test-y_pred, kde=True, bins=40)

plt.show()
# Normal distribution of residuals
# Evaluating the coefficients

df_coef = pd.DataFrame(data=lm.coef_, index=X.columns, columns=['Coefficient'])

df_coef
# From this basic analysis, 'Time on App' (per unit time) has a greater impact on yearly spending vs 'Time on Website' (per unit time)

# Length of Membership appears to have the greatest impact (per unit basis) assuming all other factors remain constant