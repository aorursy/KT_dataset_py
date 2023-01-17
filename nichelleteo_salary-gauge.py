import csv

import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline



with open('data.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["Job", "Type", "Industry", "Exp", "Size of company", "Freq of hire yrly", "Salary"])

    

    writer.writerow(["Admin", "Full-time", "Retail", "1", "100", "1", "1.8"])

    writer.writerow(["Admin", "Full-time", "Real Estate", "1", "25", "0.5", "2"])

    writer.writerow(["Admin", "Part-time", "Religious Organisation", "0", "70", "0.5", "2.3"])

    writer.writerow(["Admin", "Full-Time", "Construction", "3", "80", "1", "2.8"])

    writer.writerow(["Admin", "Temp", "Service", "0", "20", "0.25", "1.8"])

    writer.writerow(["Admin", "Part-time", "Logistics", "3", "200", "1", "1.4"])

    writer.writerow(["Admin", "Temp", "Retail", "2", "40", "0.75", "1.5"])

    writer.writerow(["Admin", "Part-time", "F&B", "3", "160", "3", "1.6"])

    writer.writerow(["Admin", "Full-Time", "Service", "3", "50", "0.25", "2"])

    

dataset = pd.read_csv('data.csv')

dataset.isnull().any()

dataset = dataset.fillna(method='ffill')

dataset.shape

dataset.describe()

# Linear regression of Exp v salary

dataset.plot(x='Exp', y='Salary', style='o')  

plt.title('Salary vs Exp')  

plt.xlabel('Exp')  

plt.ylabel('Salary')  

plt.show()



plt.figure(figsize=(15,10))

plt.tight_layout()

# seabornInstance.distplot(dataset['Salary'])

X = dataset['Exp'].values.reshape(-1,1)

y = dataset['Salary'].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) 

print(regressor.intercept_)

print(regressor.coef_)

# yearsOfExp = input('How many years of exp?')

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



plt.plot(X_test, y_pred, color='black', linewidth=2)

plt.scatter(x=X,y=y)

plt.xlabel('Exp')

plt.ylabel('Salary')

plt.show()



# Use this to see the approx salary given the size of company

# inputting = input('No of years: ')

# inputting = np.asarray(inputting, dtype='float64')

# prediction = regressor.intercept_ + regressor.coef_ * inputting

# print(prediction)
# Linear regression of size of company v salary

dataset.plot(x='Size of company', y='Salary', style='x')  

plt.title('Salary vs Size of company')  

plt.xlabel('Size of company')  

plt.ylabel('Salary')  

plt.show()



plt.figure(figsize=(15,10))

plt.tight_layout()

# seabornInstance.distplot(dataset['Salary'])

X = dataset['Size of company'].values.reshape(-1,1)

y = dataset['Salary'].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) 

print(regressor.intercept_)

print(regressor.coef_)

# yearsOfExp = input('How many years of exp?')

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



plt.plot(X_test, y_pred, color='black', linewidth=2)

plt.scatter(x=X,y=y)

plt.xlabel('Size of company')

plt.ylabel('Salary')

plt.show()



# Use this to see the approx salary given the size of company

# inputting = input('Size of company: ')

# inputting = np.asarray(inputting, dtype='float64')

# prediction = regressor.intercept_ + regressor.coef_ * inputting

# print(prediction)
#multi-linear regression of Exp/ Size of company/ frequency of hire v salary

from sklearn import linear_model

from sklearn.model_selection import train_test_split



df = pd.read_csv('data.csv')

df.head()

cdf = df[["Exp", "Size of company", "Freq of hire yrly", "Salary"]]

cdf.head()

msk = np.random.rand(len(df)) < 0.8

train = cdf[msk]

test = cdf[~msk]

regr = linear_model.LinearRegression()

x = np.asanyarray(train[["Exp", "Size of company", "Freq of hire yrly"]])

y = np.asanyarray(train[["Salary"]])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regr.fit(X_train,y_train)

print('Coefficients: ', regr.coef_)