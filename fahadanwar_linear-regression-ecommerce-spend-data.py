# Importing required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import learning_curve

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
customers = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
customers.info()
customers.head()
sns.pairplot(customers)
sns.heatmap(customers.corr(), linewidth=0.5, annot=True)
x = customers[['Time on App', 'Length of Membership']]

y = customers['Yearly Amount Spent']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 50)
lm = LinearRegression()

lm.fit(x_train, y_train)
# Function to Plot Learning curve

def plot_lc(estimator, x, y, train_sizes):

    train_sizes, train_scores, test_scores = learning_curve(lm,x,y, train_sizes = train_sizes, cv = 5,

    scoring = 'neg_mean_squared_error')

    train_scores_mean = np.mean(-train_scores, axis=1)

    train_scores_std = np.std(-train_scores, axis=1)

    test_scores_mean = np.mean(-test_scores, axis=1)

    test_scores_std = np.std(-test_scores, axis=1)



    plt.style.use('seaborn')

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')

    plt.plot(train_sizes, test_scores_mean, label = 'Validation error')

    plt.ylabel('MSE', fontsize = 14)

    plt.xlabel('Training set size', fontsize = 14)

    plt.title('Learning curve', fontsize = 18, y = 1.03)

    plt.legend()
print("Coeffs are Time on App : {0} , Length of Membership: {1}".format(lm.coef_[0], lm.coef_[1]))

print("Intercept : ",lm.intercept_)
result = lm.predict(x_test)
plt.scatter(y_test, result)

plt.xlabel("Actual values")

plt.ylabel("Predicted values")
plot_lc(lm,x,y,np.linspace(5, len(x_train), 10, dtype='int'))
print('R2 score : ',metrics.r2_score(y_test, result))

print('Variance: ',metrics.explained_variance_score(y_test,result))

print('MSE: ', metrics.mean_squared_error(y_test,result))
x = customers[['Time on App', 'Length of Membership','Avg. Session Length']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 50)
lm.fit(x_train, y_train)
print("Coeffs are Time on App : {0} , Length of Membership: {1} , Avg. Session Length: {2}".format(lm.coef_[0], lm.coef_[1], lm.coef_[2]))

print("Intercept : ",lm.intercept_)
result = lm.predict(x_test)
plt.scatter(y_test, result)

plt.xlabel("Actual values")

plt.ylabel("Predicted values")
plot_lc(lm,x,y,np.linspace(5, len(x_train), 10, dtype='int'))
print('R2 score : ',metrics.r2_score(y_test, result))

print('Variance: ',metrics.explained_variance_score(y_test,result))

print('MSE ', metrics.mean_squared_error(y_test,result))