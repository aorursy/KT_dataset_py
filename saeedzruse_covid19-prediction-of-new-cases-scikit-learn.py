import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import datetime as dt
print(os.getcwd())
dataset = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
dataset.head(4)
xLength = len(dataset.columns) - 4

X = np.arange(xLength)
dataset['Country/Region'][0:4]
analyzedContry = 'Italy'



filt3 = (dataset['Country/Region'] == analyzedContry)



y = np.transpose(dataset.loc[filt3, dataset.columns[4:]].apply(pd.Series.sum).values)

print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)

print(X_test)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()



X_train = sc_X.fit_transform(X_train.reshape(-1, 1))

X_test = sc_X.transform(X_test.reshape(-1, 1))
#Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X.reshape(-1, 1))

poly_reg.fit(X_poly, y.reshape(-1, 1))

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y.reshape(-1, 1))
# Visualising the Linear Regression results

plt.scatter(X.reshape(-1, 1), y.reshape(-1, 1), color = 'red', alpha = 0.4)

plt.plot(X.reshape(-1, 1), lin_reg.predict(X.reshape(-1, 1)), color = 'blue')

plt.title('{} Corona Infected Cases (Linear Regression)'.format(analyzedContry))



#plt.figure(figsize=(15, 15))

#plt.xticks(np.arange(xLength), pd.to_datetime(dataset.columns[4:], format ='%m/%d/%y').tolist())



plt.xlabel('Time (days)')

plt.ylabel('Number of Cases in {}'.format(analyzedContry))

plt.show()
# Visualising the Polynomial Regression results 

# Red dots are actual cases , Blue line is the fitted function

plt.scatter(X.reshape(-1, 1), y.reshape(-1, 1), color = 'red', alpha = 0.4)

plt.plot(X.reshape(-1, 1), lin_reg_2.predict(poly_reg.fit_transform(X.reshape(-1, 1))), color = 'blue')

plt.title(' {} Corona Infected Cases (Polynomial Regression)'.format(analyzedContry))

plt.xlabel('Time (days)' )

plt.ylabel('Number of Cases in {}'.format(analyzedContry))

plt.show()
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# Red dots are actual cases , Blue line is the fitted function



X_grid = np.arange(min(X.reshape(-1, 1)), max(X.reshape(-1, 1)), 0.1)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X.reshape(-1, 1), y.reshape(-1, 1), color = 'red', alpha = 0.4)

plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')

plt.title('{} Corona Infected Cases (Polynomial Regression)'.format(analyzedContry))

plt.xlabel('Time (days)')

plt.ylabel('Number of Cases in {}'.format(analyzedContry))

plt.show()
# Predicting a new result with Linear Regression

lin_reg.predict([[140]])


# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform([[140]]))
#Training the Logistic Regression model on the Training set



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X.reshape(-1, 1), y.reshape(-1, 1))
y_pred = classifier.predict(X_test)
# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))
# Visualising the Polynomial Regression results

# Red dots are actual cases , Blue line is the fitted function



plt.scatter(X.reshape(-1, 1), y.reshape(-1, 1), color = 'red', alpha = 0.4)

plt.plot(X.reshape(-1, 1), classifier.predict(X.reshape(-1, 1)), color = 'blue')



plt.title(' {} Corona Infected Cases (Logistic Regression)'.format(analyzedContry))

plt.xlabel('Time (days)' )

plt.ylabel('Number of Cases in {}'.format(analyzedContry))

plt.show()