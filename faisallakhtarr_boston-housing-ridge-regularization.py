import pandas as pd



from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression



from sklearn import metrics

from sklearn.metrics import r2_score

from numpy import sqrt



from sklearn.linear_model import Ridge
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"]

data = pd.read_csv("../input/bostonhousing/boston-housing.csv", header=None, delimiter=r"\s+", names=column_names)



print(data.head())
X = data.drop('PRICE',axis=1)

Y = data['PRICE']

print(X.head())

print(Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(X_test.shape, Y_test.shape)

print(X_train.shape, Y_train.shape)
lr = LinearRegression() 

lr.fit(X_train, Y_train)



Y_pred = lr.predict(X_test)
print('Mean absolute error : ', metrics.mean_absolute_error(Y_test,Y_pred))

print('Mean square error : ', metrics.mean_squared_error(Y_test,Y_pred))

print('R squared error', r2_score(Y_test,Y_pred))

print('RMSE', sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
rr = Ridge(alpha=0.5)

rr.fit(X_train, Y_train)



Y_predRR = rr.predict(X_test)
print('Mean absolute error : ', metrics.mean_absolute_error(Y_test,Y_predRR))

print('Mean square error : ', metrics.mean_squared_error(Y_test,Y_predRR))

print('R squared error', r2_score(Y_test,Y_predRR))

print('RMSE', sqrt(metrics.mean_squared_error(Y_test,Y_predRR)))
train_score=lr.score(X_train, Y_train)

test_score=lr.score(X_test, Y_test)



Ridge_train_score = rr.score(X_train, Y_train)

Ridge_test_score = rr.score(X_test, Y_test)
print("Linear regression train score:", train_score)

print("Linear regression test score:", test_score)

print("Ridge regression train score:", Ridge_train_score)

print("Ridge regression test score:", Ridge_test_score)