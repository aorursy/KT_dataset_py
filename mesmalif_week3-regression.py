import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
hp=pd.read_table('../input/housingdata.csv', sep=',')



hp.head()
hp.info()
hp.describe()
hps=hp[['CRIM', 'RM', 'AGE', 'DIS', 'MEDV']]
sns.pairplot(hps)

plt.show()
hps.corr()
sns.heatmap(abs(hps.corr()), vmax=1, square=True)

plt.show()
from sklearn.model_selection import train_test_split



#X=HPs[['CRIM','AGE']].values 

X=hps['RM'].values 

y=hps['MEDV'].values 



X=X.reshape(-1,1)

y=y.reshape(len(X),1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train.shape

from sklearn import linear_model

regr = linear_model.LinearRegression()



regr.fit (X_train, y_train)



# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
y_pred=regr.predict(X_test)



from sklearn.metrics import mean_squared_error



mean_squared_error(y_pred , y_test)

regr.coef_[0][0]
plt.scatter(hps['RM'], hps['MEDV'],  color='blue')

plt.plot(X_train, regr.coef_[0][0]*X_train + regr.intercept_[0], '-r')

plt.xlabel("Number of Room")

plt.ylabel("Med Value")



plt.show()