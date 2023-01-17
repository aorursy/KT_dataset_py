import numpy as np

from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV
#Load Iris dataset

iris = load_iris()

X_iris = iris.data

y_iris = iris.target



#Check the shape of data

print (X_iris.shape)

print (y_iris.shape)



#Check if sets balanced

print ('Test  1: {}, 2: {}, 3: {}'.format(np.sum(y_iris == 0), np.sum(y_iris == 1), np.sum(y_iris == 2) ) )

print ('Train 1: {}, 2: {}, 3: {}'.format(np.sum(y_iris == 0), np.sum(y_iris == 1), np.sum(y_iris == 2) ) )

#Create separate each feature array

a = X_iris[:,0]

b = X_iris[:,1]

c = X_iris[:,2]

d = X_iris[:,3]





#Scale X data

scaler = StandardScaler()

scaler.fit_transform (X_iris,y_iris)

X_scaled = scaler.transform (X_iris)





#Make modified features sets with squares of each feature

X_squares   =  np.vstack (([a**2], [b**2], [c **2], [d**2])).T



#Make modified features set with multiplied pairs of each feature

X_multi = np.vstack ((a*b, a*c, a*d, b*c, b*d, c*d)).T



#Make polynomial transformation n = 10

transform = PolynomialFeatures(10)

transform.fit_transform(X_iris)

X_poly = transform.transform(X_iris)

#Make split for original data

(X_tr_o, X_ts_o, y_tr_o, y_ts_o ) = train_test_split(X_iris, y_iris, stratify=y_iris, test_size= 0.3)



#Make split for scaled data

(X_tr_sc, X_ts_sc, y_tr_sc, y_ts_sc) = train_test_split(X_scaled, y_iris, stratify = y_iris, test_size = 0.30)



#Make split of polynomial extended features set

(X_tr_p, X_ts_p, y_tr_p, y_ts_p ) = train_test_split(X_poly, y_iris, stratify=y_iris, test_size= 0.3)



#Make split of squares of each feature

(X_tr_sq, X_ts_sq, y_tr_sq, y_ts_sq ) = train_test_split(X_squares, y_iris, stratify = y_iris, test_size = 0.3)



#Make split of  multyplied pairs of each feature

(X_tr_m, X_ts_m, y_tr_m, y_ts_m ) = train_test_split(X_multi, y_iris, stratify = y_iris, test_size = 0.3)

#Create estimator class

estimator = LogisticRegression()



#Create param grid

paramgrid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'penalty': ['l1','l2']}



#Create SearchGridCV optimizer

optimizer = GridSearchCV(estimator, paramgrid, cv=10)



#Fit it for original data

optimizer.fit(X_tr_o, y_tr_o)

predict = optimizer.best_estimator_.predict(X_ts_o)

z_o = accuracy_score(y_ts_o,predict)



#Fit it for scaled data

optimizer.fit(X_tr_sc, y_tr_sc)

predict = optimizer.best_estimator_.predict(X_ts_sc)

z_sc = accuracy_score(y_ts_sc,predict)



#Fit it for  multiplied and squared features set

optimizer.fit(X_tr_p, y_tr_p)

predict = optimizer.best_estimator_.predict(X_ts_p)

z_p = accuracy_score(y_ts_p,predict)



#Fit it for squares only of each feature

optimizer.fit(X_tr_sq, y_tr_sq)

predict = optimizer.best_estimator_.predict(X_ts_sq)

z_sq = accuracy_score(y_ts_sq,predict)



#Fit it for multiplaied pairs of features

optimizer.fit(X_tr_m, y_tr_m)

predict = optimizer.best_estimator_.predict(X_ts_m)

z_m = accuracy_score(y_ts_m,predict)
print ('Accuracy score for original: {}'.format(  z_o) )

print ('Accuracy score for scaled: {}'.format (  z_sc) )

print ('Accuracy score for polynomial: {}'.format( z_p) )

print ('Accuracy score for squares: {}'.format( z_sq) )

print ('Accuracy score for multi: {}'.format(  z_m) )
