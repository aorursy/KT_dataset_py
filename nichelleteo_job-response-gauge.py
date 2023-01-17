# Max dataset is approx 10gb

import csv

import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn import metrics

%matplotlib inline



with open('data.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["Job", "Type", "No. of Applicants", "Location (avg time to work)", "Salary", "BrandIndex"])

    

    writer.writerow(["Admin", "Full-time", "150", "30", "1.8", "5.3"])

    writer.writerow(["Cashier", "Full-time", "40", "30", "1.3", "3.2"])

    writer.writerow(["Store assistant", "Part-time", "100", "40", "1.6", "4.6"])

    writer.writerow(["Packer", "Part-Time", "70", "45", "1.5", "0.9"])

    writer.writerow(["Call centre agent", "Temp", "180", "15", "2.5", "4.2"])

    writer.writerow(["Admin", "Part-time", "140", "40", "1.8", "7.5"])

    writer.writerow(["Driver", "Full-time", "130", "20", "3.0", "2.2"])

    writer.writerow(["Marketing", "Full-time", "90", "30", "2.0", "1.6"])

    writer.writerow(["Kitchen crew", "Full-time", "90", "25", "1.4", "0.3"])

    

dataset = pd.read_csv('data.csv')

dataset.isnull().any()

dataset = dataset.fillna(method='ffill')
# Linear regression of Salary v number of applicants

dataset.plot(x='Salary', y='No. of Applicants', style='o')  

plt.title('No. of Applicants vs Salary')  

plt.xlabel('Salary')  

plt.ylabel('No. of Applicants')  

plt.show()



plt.figure(figsize=(15,10))

plt.tight_layout()

# seabornInstance.distplot(dataset['Salary'])

X = dataset['Salary'].values.reshape(-1,1)

y = dataset['No. of Applicants'].values.reshape(-1,1)



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

plt.xlabel('Salary')

plt.ylabel('No. of Applicants')

plt.show()



# Use this to see No. of applicants given the salary

# inputting = input('Salary: ')

# inputting = np.asarray(inputting, dtype='float64')

# prediction = regressor.intercept_ + regressor.coef_ * inputting

# print(prediction)
# Linear regression of avg time to get to work(proxy for location measure) v number of applicants

dataset.plot(x='Location (avg time to work)', y='No. of Applicants', style='o')  

plt.title('No. of Applicants vs Location (avg time to work)')  

plt.xlabel('Location (avg time to work)')  

plt.ylabel('No. of Applicants')  

plt.show()



plt.figure(figsize=(15,10))

plt.tight_layout()

# seabornInstance.distplot(dataset['Salary'])

X = dataset['Location (avg time to work)'].values.reshape(-1,1)

y = dataset['No. of Applicants'].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) 

# print(regressor.intercept_)

print(regressor.coef_)

# yearsOfExp = input('How many years of exp?')

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



plt.plot(X_test, y_pred, color='black', linewidth=2)

plt.scatter(x=X,y=y)

plt.xlabel('Location (avg time to work)')

plt.ylabel('No. of Applicants')

plt.show()



# Use this to see the approx number of applicants given average time required for applicants to get to that workplace

# x = print('Location (avg time to work): ')

# inputting = input()

# inputting = np.asarray(inputting, dtype='float64')

# prediction = regressor.intercept_ + regressor.coef_ * inputting

# print(prediction)
# Linear regression of BrandIndex v number of applicants

# Problem with brand index is that it is managed by 'YouGov'. Don't know exactly how it's computed

dataset.plot(x='BrandIndex', y='No. of Applicants', style='o')  

plt.title('No. of Applicants vs BrandIndex')  

plt.xlabel('BrandIndex')  

plt.ylabel('No. of Applicants')  

plt.show()



plt.figure(figsize=(15,10))

plt.tight_layout()

# seabornInstance.distplot(dataset['Salary'])

X = dataset['BrandIndex'].values.reshape(-1,1)

y = dataset['No. of Applicants'].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train) 

# print(regressor.intercept_)

print(regressor.coef_)

# yearsOfExp = input('How many years of exp?')

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



plt.plot(X_test, y_pred, color='black', linewidth=2)

plt.scatter(x=X,y=y)

plt.xlabel('BrandIndex')

plt.ylabel('No. of Applicants')

plt.show()
#multi linear regression to determine number of applicants based on avg time to work/ location + brandindex + salary

df = pd.read_csv('data.csv')



X = df[["Location (avg time to work)", "Salary", "BrandIndex"]]

y = df["No. of Applicants"]



regr = linear_model.LinearRegression()

regr.fit(X, y)



# Use this version to be able to input your own numbers

"""

print("Location (avg time to work): ")

inputt = np.asarray(input(), dtype='float64')

print("Salary: ")

inputting = np.asarray(input(), dtype='float64')

print("BrandIndex: ")

inputs = np.asarray(input(), dtype='float64')

predicted = regr.predict([[30, inputting, inputs]])

print('Predicted no. of applications')

print(predicted)

"""

print("Location (avg time to work): 30")

print("Salary: 1.6")

print("BrandIndex: 2.1")

predicted = regr.predict([[30, 1.6, 2.1]])

print('Predicted no. of applications')

print(predicted)