# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory

# print all the file/directories present at the path

import os

print(os.listdir("../input/"))
# importing the dataset

dataset = pd.read_csv('../input/50_Startups.csv')

dataset.head()
dataset.info()
dataset.isnull().sum()

# State is a categorical variable with 3 different values possible.

# California is with top frequenecy of 17

dataset.iloc[:,3].describe()

# matrix of features as X and dep variable as Y (convert dataframe to numpy array)

X = dataset.iloc[:,:-1].values          #R&D spend, Administration, Marketing Spend, State

Y = dataset.iloc[:,-1].values           #Profit
# Encoding Categorical variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

en = LabelEncoder()

X[:,3] = en.fit_transform(X[:,3])

oh = OneHotEncoder(categorical_features=[3])

X = oh.fit_transform(X)                                   #type(X)==sparse matrix



# converting from matrix to array

X = X.toarray()



# Dummy variable trap ---- Removing one dummy variable 

X = X[:,1:]
# R&D Spend vs Profit

from matplotlib import pyplot

pyplot.scatter(dataset.iloc[:,0:1].values,dataset.iloc[:,4:5])

pyplot.title('R&D Spend vs Profit')
# Administration vs Profit

from matplotlib import pyplot

pyplot.scatter(dataset.iloc[:,1:2].values,dataset.iloc[:,4:5])

pyplot.title('Administration vs Profit')
# Marketing spend vs Profit

from matplotlib import pyplot

pyplot.scatter(dataset.iloc[:,2:3].values,dataset.iloc[:,4:5])

pyplot.title('Marketing spend vs Profit')

# statsmodels is a Python module that provides classes and functions for the estimation of 

# many different statistical models, as well as for conducting statistical tests, and 

# statistical data exploration.



import statsmodels.api as sm





# Lib statmodels.formula.api doesn't take b0 in account as constant. So somehow we need to add one 

# indep var in our matrix of features X. So, if we add a column with '1'(s) in our matrix - 

# statmodels lib would understand that this is X0 associated with b0.

#----->  y = b0X0+b1*x1+b2*x2......+bn*xn            ////Where X0 = 1



# Appending an array of 1's as first column to matrix of features e.g. X

X = np.append(np.ones((50,1),dtype=int),X,axis=1)







X.shape
# Now we shall create another matrix of features which will be our optimal one e.g. (X_opt). It will contain only 

# the independant var which have high impact on calculation the  the profit e.g. dependant vector using 

# the Backward elimination method.



# ----> Let's consider Significance Level = 0.05 and first create the model with all predictors.



X_opt = X[:, [0,1,2,3,4,5]]

reg_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
reg_OLS.summary()
# Now we select the Predictor with Highest P value > SL and eliminate that predictor and fit the model 

# with rest of predictor. Repeat the step till we have all predictors with P<SL.



# ----> P value is statistical measure that helps to determine if null hypothesis is correct or not.



# -----> x2 e.g. feature at thrid position has highest P = 0.99 



X_opt = X[:, [0,1,3,4,5]]

regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

regressor_OLS.summary()
# -----> Eliminate x1 from X_opt e.g. feature at second position has highest P = 0.94 



X_opt = X_opt[:, [0,2,3,4]]

regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

regressor_OLS.summary()
# -----> Eliminate x2 from X_opt e.g. feature at thrid position has highest P = 0.602



X_opt = X_opt[:, [0,1,3]]

regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

regressor_OLS.summary()
# -----> Eliminate x2 from X_opt e.g. feature at thrid position has highest P = 0.06



X_opt = X_opt[:, [0,1]]

regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

regressor_OLS.summary()
# Looking at the above OLS summary ---> Apart from the constant  (x0); the maximum impacting predictor is

# R&D spent (e.g. X1 in X_opt)



# ----> printing the values of coffiecients



# ----> final MLR equation is 

# profit = 4.90328991e+04 * 0.854291371(R&D Spend)



regressor_OLS.params