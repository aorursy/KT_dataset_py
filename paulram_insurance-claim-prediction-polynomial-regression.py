#import the basic libraries that are mostly used to build any machine learning algorithm

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
#Read the input excel file

ins=pd.read_csv('../input/insurance/insurance.csv')

ins.head()

#check for any NaN

print(ins.isna().sum())

print(ins.info())

ins.describe()

#Lets get dummies for the categorical variables.

catcol=['sex','smoker','region']

ins1=pd.get_dummies(ins,columns=catcol)

ins1.columns
#Lets see now if there is any outlier on the independent variable

#Lets import the seaborn library to plot the outlier. 

import seaborn as sns

sns.boxplot(x=ins1['bmi'])

sns.boxplot(x=ins1['age'])
#import the stats fucntion from the scipy library

from scipy import stats

z = np.abs(stats.zscore(ins1))

#x=stats.zscore(ins1)

print(z)
print(np.where(z>3))
ins1_o=ins1[(z<3).all(axis=1)]

ins1_o
#Import the libraries

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

%matplotlib inline
# #Lets define the dependent and independent variables.

X=ins1[['age','bmi','sex_female','sex_male','smoker_no','smoker_yes','region_northeast','region_northwest','region_southeast','region_southwest']]

y=ins1['charges']

# plt.figure(figsize=(15,10))

#plt.tight_layout()

sns.distplot(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

LRM = LinearRegression()  

LRM.fit(X_train, y_train)
#Lets check the best intercept and the coefficients

print('Intercept:',LRM.intercept_)

print('Coefficients for different independent variables:',LRM.coef_)
coeff_df = pd.DataFrame(LRM.coef_, X.columns, columns=['Coefficient'])  

coeff_df
#Lets predict using the trained model and compare it against the actuals

y_pred = LRM.predict(X_test)

compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

compare.head(10) #since it is 250+ rows, lets see only the first 10 rows
compare.head(25).plot(kind='bar',figsize=(10,8))
print('Mean Absolute Error without removing the outliers:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error without removing the outliers:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error without removing the outliers:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R2 value without removing the outliers:', r2_score(y_test, y_pred))
X1=ins1[['age','bmi','sex_female','sex_male','smoker_no','smoker_yes']]

y1=ins1['charges']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)

LRM1 = LinearRegression()  

LRM1.fit(X1_train, y1_train)

#Lets check the best intercept and the coefficients

print('Intercept:',LRM1.intercept_)

print('Coefficients for different independent variables:',LRM1.coef_)

coeff_df1 = pd.DataFrame(LRM1.coef_, X1.columns, columns=['Coefficient'])  

coeff_df1



#Lets predict using the trained model and compare it against the actuals

y1_pred = LRM1.predict(X1_test)

compare1 = pd.DataFrame({'Actual': y1_test, 'Predicted': y1_pred})

compare1.head(10) #since it is 250+ rows, lets see only the first 10 rows

compare1.head(25).plot(kind='bar',figsize=(10,8))

print('Mean Absolute Error without removing the outliers and without regions:', metrics.mean_absolute_error(y1_test, y1_pred))  

print('Mean Squared Error without removing the outliers and without regions:', metrics.mean_squared_error(y1_test, y1_pred))  

print('Root Mean Squared Error without removing the outliers and without regions:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))



print('R2 value without removing the outliers and without regions:', r2_score(y1_test, y1_pred))
X2=ins1_o[['age','bmi','sex_female','sex_male','smoker_no','smoker_yes']]

y2=ins1_o['charges']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

LRM2 = LinearRegression()  

LRM2.fit(X2_train, y2_train)

#Lets check the best intercept and the coefficients

print('Intercept:',LRM2.intercept_)

print('Coefficients for different independent variables:',LRM2.coef_)

coeff_df1 = pd.DataFrame(LRM1.coef_, X1.columns, columns=['Coefficient'])  

coeff_df1



#Lets predict using the trained model and compare it against the actuals

y2_pred = LRM2.predict(X2_test)

compare2 = pd.DataFrame({'Actual': y2_test, 'Predicted': y2_pred})

compare2.head(10) #since it is 250+ rows, lets see only the first 10 rows

compare2.head(25).plot(kind='bar',figsize=(10,8))

print('Mean Absolute Error with outliers removed:', metrics.mean_absolute_error(y2_test, y2_pred))  

print('Mean Squared Error with outliers removed::', metrics.mean_squared_error(y2_test, y2_pred))  

print('Root Mean Squared Error with outliers removed:', np.sqrt(metrics.mean_squared_error(y2_test, y2_pred)))

print('R2 value with outliers removed:', r2_score(y2_test, y2_pred))
from sklearn.preprocessing import PolynomialFeatures

XP=ins1_o[['age','bmi','sex_female','sex_male','smoker_no','smoker_yes']]

yP=ins1_o['charges']

XP_train, XP_test, yP_train, yP_test = train_test_split(XP, yP, test_size=0.2, random_state=0)

polynomial_features= PolynomialFeatures(degree=4)

X_poly = polynomial_features.fit_transform(XP_train)

XP_poly_test = polynomial_features.fit_transform(XP_test)

model = LinearRegression()

model.fit(X_poly, yP_train)

y_poly_pred = model.predict(XP_poly_test)

rmse = np.sqrt(mean_squared_error(yP_test,y_poly_pred))

r2 = r2_score(yP_test,y_poly_pred)

print('Root Mean Square Value through Polynominal Regression',rmse)

print('R2 Score through Polynominal Regression:', r2)
compare_poly = pd.DataFrame({'Actual': yP_test, 'Predicted': y_poly_pred})

compare_poly.head(10) #since it is 250+ rows, lets see only the first 10 rows

compare_poly.head(25).plot(kind='bar',figsize=(10,8))