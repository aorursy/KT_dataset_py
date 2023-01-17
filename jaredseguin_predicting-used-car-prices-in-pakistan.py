import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/used-cars-data-pakistan/OLX_Car_Data_CSV.csv", header=0,encoding = 'unicode_escape')
df.head(5)
df.shape
df.isnull().sum()


df2 = df.dropna(axis = 0)
df2.isnull().sum()
df2.count()
not_num = df2.select_dtypes(include = ['object']).columns

df2[not_num].head()
df3 = df2[df2.Condition != 'New']



# Above we created a new dataframe by dropping all the New car rows. I am only interested in the Used car market for this analysis. 



df3['Price'].hist(bins=50)

plt.ylabel('Frequency')

plt.xlabel('Price of listing')

plt.title('Histogram of prices')



df3['Price'] = np.log(df3.Price)

df3['Price'].hist(bins=50)

plt.ylabel('Frequency')

plt.xlabel('Price of listing')

plt.title('Histogram of prices')


df3['Transaction Type'].describe()
df3 = df3.drop("Condition", axis=1)

df3 = df3.drop("Transaction Type", axis=1)

not_num = df3.select_dtypes(include = ['object']).columns

df3[not_num].head()
dummies = pd.get_dummies(df3[not_num], drop_first = True)

dummies.head()


df3[not_num].describe()


df4 = df3.drop(not_num, axis = 1)

df4 = pd.merge(df4, dummies, left_index = True, right_index = True)

df4.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X = df4.drop('Price', axis = 1)

y = df4['Price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

Reg = LinearRegression()

Reg.fit(X_train, y_train)

y_pred = Reg.predict(X_test)
from sklearn.metrics import r2_score

print('R^2 for test data:')

print(r2_score(y_test, y_pred))



from sklearn.metrics import mean_squared_error

print('RMSE for test data:')

print(np.sqrt(mean_squared_error(y_test, y_pred)))
from sklearn.ensemble import RandomForestRegressor as RFR

ForestReg = RFR() #RFR(n_estimators = 150, random_state = 3, max_depth = 15) performed better



ForestReg.fit(X_train, y_train)



y_F_pred = ForestReg.predict(X_test)
print('R^2 for test data:')

print(r2_score(y_test, y_F_pred))

print('RMSE for test data:')

print(np.sqrt(mean_squared_error(y_test, y_F_pred)))
plt.scatter(y_test, y_pred)

plt.ylabel('Linear Regression Predicted Price')

plt.xlabel('Actual Price')

plt.title('Predicted vs Actual Prices for Linear Regression')
plt.scatter(y_test, y_F_pred)

plt.ylabel('RandomForest Predicted Price')

plt.xlabel('Actual Price')

plt.title('Predicted vs Actual Prices for Random Forest')
sns.distplot(y_test-y_pred)

plt.title('Residual PDF for Linear Regression')
sns.distplot(y_test-y_F_pred)

plt.title('Residual PDF for RandomForest')