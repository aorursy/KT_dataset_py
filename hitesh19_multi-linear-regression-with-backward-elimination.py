# Introduction 
 # Linear regression  is used to predict the value of outcome variable  Y, based on one or more than input variable X.
 # The aim is to establish a linear realtionship between the predictor variable and tha respones  variable
 # so we can use this formula to estimate the vlaue of the esponse Y. when only the predictor (X) value are Known.
 # Tools : Python programming language and using some library like numpy, pandas,sklearn ,matplotlib,statmodel.

# Bojective : prediction the profit of Startup.
# X variables are R&D Spend,Administration,Marketing, State.
# Y variable is profit.
# by using the Multilinear Regression with python.
# import the required library packages 
import numpy as np # basically use for  computing the scientific  calculation.
import pandas as pd # basically use for data manipulation.\
import statsmodels.formula.api as sm # Basically use for some statistical solutions.

data = pd.read_csv("../input/50_Startups.csv")
# display the below row of data
data.head()
# display the structure the data
data.info()
# dispaly the describe of data exclude the categorical variable
data.describe()
# display the categorical variables
data['State'].head()
# unique variables in categorical variables
data['State'].unique()
X = data.iloc[:, :-1].values
y = data.iloc[:,4].values
print (X)
print (y)
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#Avoiding dummy variable trap
X = X[:, 1:]

#Splitting into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#Fitting multiple linear regression to trainingset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
model1=sm.OLS(y_train,X_train)
result = model1.fit()
result.summary()
#Building the optimla model using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
