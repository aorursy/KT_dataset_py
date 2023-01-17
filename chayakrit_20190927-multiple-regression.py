# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/50-startups/50_Startups.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values
# Try running this code, you'll get a warning message. 



# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #Change text into number

onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [3] but can be [0, 1, 3])

    remainder='passthrough'                         # Leave the rest of the columns untouched

)



X = np.array(ct.fit_transform(X), dtype=np.float)
# Avoiding the dummy variable trap

#Remove the first column (Column 0th) -> Take values from column 1 onward

X = X[:, 1:] 
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#Predicting the Test Set Results

y_pred = regressor.predict(X_test)
# Building the optimal model using Backward Elimination

import statsmodels.api as sm

X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis=1) #Add 1 to the 1st column
print(X)
#X_opt is the X that contains only the optimal variables

X_opt = X[:,[0,1,2,3,4,5]] # Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
# The number shows that x2 has the highest P-value (0.99), so we remove it.

X_opt = X[:,[0,1,3,4,5]] # Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
# The number shows that x1 has the highest P-value (0.94), so we remove it.

X_opt = X[:,[0,3,4,5]] # Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
# The number shows that x4 has the highest P-value (0.6), so we remove it.

X_opt = X[:,[0,3,5]] # Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
# The number shows that x5 has the highest P-value (0.06, so we remove it.

X_opt = X[:,[0,3]] # Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()