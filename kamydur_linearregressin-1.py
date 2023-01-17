import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/Salary_Data.csv')

data.head(3)
data.info()
X = data.iloc[:,0].values

y = data.iloc[:,1].values



print(X.shape)

print(y.shape)
X = X.reshape(-1,1)

X.shape
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state = 17)

X_train.shape
from sklearn.linear_model import LinearRegression



lnr_clf = LinearRegression()



lnr_clf.fit(X_train,y_train)
y_pred = lnr_clf.predict(X_test)

y_pred
#plot for train data

plt.scatter(X_train,y_train,color ='red')

plt.plot(X_train,lnr_clf.predict(X_train),color = 'green')



plt.xlabel('Year of exp')

plt.ylabel('salary')

plt.show()
#plot for test data



plt.scatter(X_test,y_test,color ='red')

plt.plot(X_test,lnr_clf.predict(X_test),color ='green')

plt.xlabel('Year of exp')

plt.ylabel('salary')

plt.show()

print('Co-efficient = ', lnr_clf.coef_)
MSE = np.mean((lnr_clf.predict(X_test) - y_test)**2)

print('MSE=',MSE)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test,y_pred)



print('MSE=',mse)
lnr_clf.score(X_test,y_test)