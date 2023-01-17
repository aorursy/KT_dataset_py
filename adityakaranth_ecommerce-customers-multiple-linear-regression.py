import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

matplotlib.rcParams['figure.figsize'] = (10,6)



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
# Reading csv file to customers

customers = pd.read_csv('../input/ecommerce/Ecommerce Customers')
customers.head()
customers.info()
customers.describe()
# Get numeric columns

num_cols = customers.select_dtypes(include=[np.number]).columns.values

print('Numeric cols :',num_cols)
# Get Categorial columns

cat_cols = customers.select_dtypes(exclude=[np.number]).columns.values

print('Categorical cols :',cat_cols)
# Check for missing data

customers.isnull().sum()
sns.heatmap(customers.corr(), annot=True, cmap='viridis')
# As from above heatmap we can see 'Length of Membership' have high correlation with 'Yearly Amount Spent'

sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=customers, s=20)
# Lets see how a linear model fits the data

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
# Comparing the Time on Website and Yearly Amount Spent columns to check the correlation make sense

# As more time on site, more money spent.

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers, s=20)
# Comparing the Time on App and Yearly Amount Spent columns to check the correlation make sense

# As more time on site, more money spent.

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers, s=20)
# Visualizing Multivariate relations via Pairplot

sns.pairplot(customers)
# Lets verify the categorical columns again

print(cat_cols)
# We can either Feature Engineer these columns or drop them

customers[cat_cols]
# Dropping above 'cat_cols'

customers.drop(cat_cols,axis=1,inplace=True)

display(customers)
num_cols
# All num_cols are features and 'Yearly Amount Spent' is the target

X = customers.drop('Yearly Amount Spent', axis=1)

y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# Model Training

lr = LinearRegression()

lr .fit(X_train,y_train)
# Scatter plot of y_true and predicted values

predicted = lr.predict(X_test)

plt.scatter(y_test,predicted)

plt.xlabel('y_test')

plt.ylabel('Predicted')
# Evaluation of Regression model

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
# The coefficients of the model

print('Coefficients: ', lr.coef_)
# Residuals

# Plotting a histogram of the residuals and it looks normally distributed

sns.distplot((y_test-predicted),bins=50)
# Let's see if we can interpret the coefficients at all to get an idea

coeffecients = pd.DataFrame(lr.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

display(coeffecients)