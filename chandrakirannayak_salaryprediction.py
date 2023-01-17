import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")

df.head()
df.info()
df.describe()
#check for null values



df.isnull().sum()
plt.boxplot(df.YearsExperience)

plt.show()
plt.boxplot(df.Salary)

plt.show()
def normalize(x):

    min = x.min()

    max = x.max()

    y = (x-min)/(max-min)

    return y
df1 = normalize(df)

df1.describe()
plt.boxplot(df1.YearsExperience)

plt.show()
plt.boxplot(df1.Salary)

plt.show()
sns.pairplot(df1, x_vars='YearsExperience', y_vars='Salary')
sns.pairplot(df1, x_vars='YearsExperience', y_vars='Salary', height=5)
X = df1.Salary



X.head()
y = df1.YearsExperience



y.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,random_state=30)
X_train=X_train[:, np.newaxis]

X_test=X_test[:, np.newaxis]
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression



lr=LinearRegression()



lr.fit(X_train,y_train)
print(lr.coef_)

print(lr.intercept_)
y_pred = lr.predict(X_test)
c = [i for i in range(1,10,1)]

fig=plt.figure()

plt.plot(c,y_test,color='blue', linewidth=2.5, linestyle='-')

plt.plot(c,y_pred,color='red', linewidth=2.5, linestyle='-')

fig.suptitle('Actual v/s Predicted')

#plt.xlabel('YearsExperience')

#plt.ylabel
from sklearn.metrics import mean_squared_error, r2_score



mse=mean_squared_error(y_test,y_pred)

r2_square = r2_score(y_test,y_pred)
print('Mean Squared Error:', mse)

print('r squarevalue:', r2_square)