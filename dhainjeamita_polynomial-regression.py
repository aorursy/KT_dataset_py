import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
plt.style.use('seaborn-colorblind')
plt.figure(figsize=(7,7))
from sklearn.datasets import make_friedman1
plt.title('Complex regression problem with one input variable')
X_F1, y_F1 = make_friedman1(n_samples = 100,n_features = 7, random_state=0)
plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50, c='purple')
plt.show()
from sklearn.linear_model import LinearRegression, Ridge
print ("\nLINEAR REGRESSION ---- ")
X_train, X_test, y_train, y_test = train_test_split(X_F1,y_F1, random_state=0)
LinearReg = LinearRegression().fit(X_train,y_train)

print ("Intercept for the Linear model - {:.3f}".format(LinearReg.intercept_))
print ("Coefficents for the Linear model - {}".format(LinearReg.coef_))
print ("Training Score = {:.3f}".format(LinearReg.score(X_train,y_train)))
print ("Testing Score = {:.3f}".format(LinearReg.score(X_test,y_test)))

print ("\nPOLYNOMIAL REGRESSION WITH DEGREE = 2 ----")
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_scale = poly.fit_transform(X_F1)

X_train, X_test, y_train, y_test = train_test_split(X_train_scale,y_F1, random_state=0)
PolyReg = LinearRegression().fit(X_train,y_train)

print ("Intercept for the Polynomial model - {:.3f}".format(PolyReg.intercept_))
print ("Coefficents for the Polynomial model - {}".format(PolyReg.coef_))
print ("Training Score = {:.3f}".format(PolyReg.score(X_train,y_train)))
print ("Testing Score = {:.3f}".format(PolyReg.score(X_test,y_test)))

print ("\nAPPLICATION OF POLYNOMIAL REGRESSION MAY LEAD TO OVER-FITTING \n HENCE LETS APPLY RIDGE REGRESSION")

ridgeReg = Ridge().fit(X_train,y_train)

print ("Intercept for the Polynomial 2 + Ridge model - {:.3f}".format(PolyReg.intercept_))
print ("Coefficents for the Polynomial 2 + Ridge model - {}".format(PolyReg.coef_))
print ("Training Score = {:.3f}".format(ridgeReg.score(X_train,y_train)))
print ("Testing Score = {:.3f}".format(ridgeReg.score(X_test,y_test)))

