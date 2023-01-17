import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/random-salary-data-of-employes-age-wise/Salary_Data.csv')
df.head()
X = df[['YearsExperience']]
Y = df[['Salary']]
plt.scatter(X, Y, color = 'blue')
plt.xlabel('YearExperience')
plt.ylabel('Salary')
plt.show()
reg = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print('Coefficients: ', reg.coef_)
print('Intercept: ', reg.intercept_)
plt.scatter(X_train, Y_train, color = 'blue')
plt.plot(X_train, reg.coef_*X_train + reg.intercept_, '-r')
plt.xlabel('YearExperience')
plt.ylabel('Salary')
plt.show()
plt.scatter(X_test, Y_test, color = 'blue')
plt.plot(X_test, Y_pred, '-r')
plt.xlabel('YearExperience')
plt.ylabel('Salary')
plt.show()
results = pd.DataFrame({
    'Actual': np.array(Y_test).flatten(),
    'Predicted': np.array(Y_pred.round(decimals = 2)).flatten(),
})

#results = pd.DataFrame(results)
results
from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(Y_pred - Y_test)))
print("Mean square error: %.2f" %np.mean((Y_pred - Y_test))**2)
print("R2-score: %.2f" % r2_score(Y_pred , Y_test))
results= results.astype('float')
df1 = results.head(8)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
example = np.array([[1.2],[2.3]])
example = example.reshape(2,-1)
prediction = reg.predict(example)
prediction
import random as rd
reg = linear_model.LinearRegression()
i = rd.sample(range(50),10)
x,a,b = [],[],[]
for e in i:
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = e)
    reg.fit(X_train,Y_train)
    Y_pred = reg.predict(X_test)
    x.append(e)
    a.append(reg.coef_.round(decimals = 2))
    b.append(reg.intercept_.round(decimals = 2))
    rand_check = pd.DataFrame({
        'random_state': np.array(x).flatten(),
        'Coefficients': np.array(a).flatten(),
        'Intercept': np.array(b).flatten(),
    })
rand_check