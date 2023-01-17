import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 



from sklearn.linear_model import LinearRegression 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor



from sklearn import preprocessing

from sklearn.model_selection import train_test_split



bitcoin = pd.read_csv('bitcoin_merged.csv')

bitcoin = bitcoin[::-1]

bitcoin.head()



X = np.array(bitcoin[['Open', 'High', 'Low']])

X = preprocessing.scale(X)

y = np.array(bitcoin['Close'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)

dtr_pred = dtr.predict(X_test)

print('Decision Tree Regressor: {}'.format(dtr.score(X_test, y_test)))



abdtr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 100)

abdtr.fit(X_train, y_train)

abdtr_pred = abdtr.predict(X_test)

abdtr_score = abdtr.score(X_test, y_test)

print('Ada Boost - Decision Tree Regressor: {}'.format(abdtr_score))



rfr = RandomForestRegressor(n_estimators = 300)

rfr.fit(X_train, y_train)

rfr_pred = rfr.predict(X_test)

print('Random Forest Regressor: {}'.format(rfr.score(X_test, y_test)))



linreg = LinearRegression()

linreg.fit(X_train, y_train)

lr_pred = linreg.predict(X_test)

linreg_score = linreg.score(X_test, y_test)

print('Linear Regression: {}'.format(linreg_score))



for k in ['linear','poly','rbf','sigmoid']:

    svr = SVR(kernel=k)

    svr.fit(X_train, y_train)

    svr_score = svr.score(X_test, y_test)

    print('SVR {}: {}'.format(k, svr_score))



knn = KNeighborsRegressor()

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_score = knn.score(X_test, y_test)

print('KNN Regressor: {}'.format(knn_score))
