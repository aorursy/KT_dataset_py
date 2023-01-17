import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

ToyotaCorolla = pd.read_csv("../input/ToyotaCorolla.csv",encoding= 'unicode_escape')

ToyotaCorolla.shape
ToyotaCorolla.columns
ToyotaCorolla.describe()
dataframe_toyota = ToyotaCorolla.drop(['Id', 'Model',  'Mfg_Month', 'Mfg_Year', 

       'Fuel_Type',  'Met_Color', 'Color', 'Automatic',  

       'Cylinders',   'Mfr_Guarantee',

       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',

       'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',

       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',

       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',

       'Radio_cassette', 'Tow_Bar'], axis =1)
dataframe_toyota.shape
dataframe_toyota .info()
dataframe_toyota.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,10))

plt.tight_layout()

sns.distplot(dataframe_toyota ['Price'])
plt.figure(figsize=(15,10))

plt.tight_layout()

sns.distplot(dataframe_toyota['Weight'])
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
X = dataframe_toyota.iloc[:,dataframe_toyota.columns != 'Price']

Y = dataframe_toyota.iloc[:, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
toyota = linear_model.LinearRegression()

toyota.fit(X_train, Y_train)
coeff_df = pd.DataFrame(toyota.coef_, X.columns, columns=['Coefficient'])

print(coeff_df)
y_pred = toyota.predict(X_test)
df_tot = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})

df_tot.head(20)
df_tot.head(17).plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
# Root Mean Squared Deviation

rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))      

r2_value = r2_score(Y_test, y_pred) 

print("Intercept: \n", toyota.intercept_)

print("Root Mean Square Error \n", rmsd)

print("R^2 Value: \n", r2_value)
#  Here the R² value is 0.59955, which shows the model is almost accurate and can make good predictions. 

# R² value can range from 0 to 1. As the R² value is not  close to 1, the model will not make better predictions.

# We will identfy which features are not perform well and eliminate from computations
# Beta0 has x0=1. Add a column of for the the first term of the #MultiLinear Regression equation.

# import statsmodels.formula.api as sm

import statsmodels.regression.linear_model as sm1

#The 0th column contains only 1 in each 50 rows

X= np.append(arr = np.ones((1436,1)).astype(int), values = X, axis=1) 

X_opt= X[:, [1,2,3,4,5,6,7,8]]

#Optimal X contains the highly impacted independent variables

#OLS: Oridnary Least Square Class. endog is the dependent variable,

#exog is the number of observations

regressor_OLS=sm1.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()
# Beta0 has x0=1. Add a column of for the the first term of the 

# MultiLinear Regression equation.

# import statsmodels.formula.api as sm

import statsmodels.regression.linear_model as sm1

#The 0th column contains only 1 in each 50 rows

X= np.append(arr = np.ones((1436,1)).astype(int), values = X, axis=1) 

X_opt= X[:, [2,3,4,5,6,7,8]]

#Optimal X contains the highly impacted independent variables

#OLS: Oridnary Least Square Class. endog is the dependent variable,

#exog is the number of observations

regressor_OLS=sm1.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()
# Beta0 has x0=1. Add a column of for the the first term of the #MultiLinear Regression equation.

# import statsmodels.formula.api as sm

import statsmodels.regression.linear_model as sm1

#The 0th column contains only 1 in each 50 rows

X= np.append(arr = np.ones((1436,1)).astype(int), values = X, axis=1) 

X_opt= X[:, [1,2,3,4,5,7]]

#Optimal X contains the highly impacted independent variables

#OLS: Oridnary Least Square Class. endog is the dependent variable,

#exog is the number of observations

regressor_OLS=sm1.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()
# Beta0 has x0=1. Add a column of for the the first term of the #MultiLinear Regression equation.

# import statsmodels.formula.api as sm

import statsmodels.regression.linear_model as sm1

#The 0th column contains only 1 in each 50 rows

X= np.append(arr = np.ones((1436,1)).astype(int), values = X, axis=1) 

X_opt= X[:, [1,2,4,5,7,8]]

#Optimal X contains the highly impacted independent variables

#OLS: Oridnary Least Square Class. endog is the dependent variable,

#exog is the number of observations

regressor_OLS=sm1.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()
# Beta0 has x0=1. Add a column of for the the first term of the #MultiLinear Regression equation.

# import statsmodels.formula.api as sm

import statsmodels.regression.linear_model as sm1

#The 0th column contains only 1 in each 50 rows

X= np.append(arr = np.ones((1436,1)).astype(int), values = X, axis=1) 

X_opt= X[:, [2,3,4,5,7,8]]

#Optimal X contains the highly impacted independent variables

#OLS: Oridnary Least Square Class. endog is the dependent variable,

#exog is the number of observations

regressor_OLS=sm1.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()
# The highest impact variable is Gears , HP and Weights are impact on prediction of Toyota car  price ...
df_tot = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})

df_tot.head(20)
df_tot.to_csv('Car Price Prediction.csv',index=0)