import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import qqplot

import statsmodels.api as sm

from scipy.stats import shapiro

%matplotlib inline
houses = pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
houses.info()
houses.head()
houses.describe()
houses['city'].unique() ### No problem on cities names.
rio = houses[houses['city'] == "Rio de Janeiro"]
rio['animal'].unique()   ## Boolean variable, so we are going to transform this text variable into a dummy variable
rio.loc[:,'animal'] = rio['animal'].apply(lambda x: 1 if x == 'acept' else 0) # if 1 -> place accepts animals.
rio.head()
rio.loc[:, 'furniture'].unique()   # Same happens.That said we can make that column a dummy variable. 

                                   # 1 is going to represent furnished places. 
rio.loc[:, 'furniture'] = rio['furniture'].apply(lambda x: 1 if x == 'furnished' else 0)
rio.head()
rio['floor'].unique() # We need to deal with the '-' value.
rio['floor'].value_counts() # We still get a lot of data points if we drop those rows in which we have the

                            # "-" value for the floor variable, so we might as well drop them.
bye = rio[rio['floor'] == "-"].index ## Getting the indexes of the rows which have the value "-" for the floor variable
rio = rio.drop(bye)
rio['floor'] = rio['floor'].apply(lambda x: int(x))
rio.head()
rio.info()
rio = rio.drop(axis = 1, columns = ['property tax (R$)', 'total (R$)', 'hoa (R$)', 'fire insurance (R$)'])
plt.figure(figsize = (15,8))

sns.heatmap(rio.corr(), annot = True) ## Floor and animal variables seems not quite related to anything
sns.pairplot(rio)

plt.tight_layout()
plt.figure(figsize = (15,6))





plt.subplot(1, 2, 1)

plt.title("Comparison between the standard distriution and Log10 transformed distribution")

sns.distplot(a = rio['rent amount (R$)'])



plt.subplot(1, 2, 2)

sns.distplot(a = np.log(rio['rent amount (R$)']+1))   



plt.figure(figsize = (15,6))





plt.subplot(1, 2, 1)

plt.title("Comparison between the standard distriution and Log10 transformed distribution")

sns.distplot(a = rio['floor'])



plt.subplot(1, 2, 2)

sns.distplot(a = np.log(rio['floor']+1))
newOrder = ['city','area','rooms','bathroom','parking spaces','floor','rent amount (R$)', 'furniture', 'animal']

rio = rio[newOrder]

rio = rio.drop(labels = 'city', axis = 1)

rio.head()     #Changing order of columns in order to separate continous variables and dummy variables
rioCont, rioDummy = rio.loc[:, 'area':'rent amount (R$)'], rio.loc[:, 'furniture':'animal'] 
rioCont = np.log(rioCont + 1)
newRio = pd.concat([rioCont, rioDummy], axis = 1, join = 'inner')

newRio.head()
plt.figure(figsize = (10,6))

sns.heatmap(newRio.corr())
sns.pairplot(newRio)
sns.distplot(a = newRio['parking spaces'])  #Parking spaces is still messed up =( 
sns.lmplot(data = newRio, x = 'area', y = 'rent amount (R$)')  # Data suggests strong linear relation between the rent values and the area
# First we create a simple function for getting the VIFs for each feature of the dataset



def vif(dataframe, add_intercept = True):

    if add_intercept == True:

        dataframe = sm.add_constant(dataframe)

        

    for i in dataframe.columns:

        y = dataframe[[i]]

        x = dataframe.drop(labels = i, axis = 1)

        model = sm.OLS(y, x)

        results = model.fit()

        rSquared = results.rsquared

        vifValue = round(1/(1 - rSquared), 2)



        print('---------------------------------------------------------------------------------------------')

        print("The regression of the independent variable ", str.upper(i), " returns a R squared value of: ", rSquared)

        print('\nThat said, the VIF for this variable is: ', vifValue)

        
vif(rio)   # VIFs are low, so we can assume that multicolinearity is not a problem (we can ignore the constant VIF).
from sklearn.model_selection import train_test_split
list(newRio.columns)



newCols = ['area',

 'rooms',

 'bathroom',

 'parking spaces',

 'floor',

 'furniture',

 'animal','rent amount (R$)']



newRio = newRio[newCols]

newRio.head()                  #Changing order of cols to put rent value in the end
X = newRio.loc[:, 'area':'animal']

y = newRio['rent amount (R$)']



print(X.head())

print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
import statsmodels.api as sm       # Statsmodels for the regression summary
X_constant = sm.add_constant(X_train) 
model = sm.OLS(y_train, X_constant).fit()

predictions = model.predict(X_constant) 
print_model = model.summary()    

print(print_model)
X_constant = X_constant.drop(['parking spaces'], axis = 1) 
model = sm.OLS(y_train, X_constant).fit()

predictions = model.predict(X_constant) 

print_model = model.summary()    

print(print_model)
#Evaluating Serial Autocorrelation

residuals = y_train - predictions

sns.scatterplot(x = predictions, y = residuals)

plt.title("Fitted values x Residuals")

plt.ylabel("residuals")

plt.xlabel("fitted values")



#Residuals seem random, no signs of autocorrelation.
#Evaluating residuals distribution

sns.distplot(residuals)

plt.title("Residuals Distribuition")

plt.xlabel("Residuals")



# Distribution seems roughly normal, so we are fine.
X_test_no_parking = X_test.drop(axis = 1, labels = 'parking spaces')
X_test_no_parking_constant = sm.add_constant(X_test_no_parking)
forecasting = model.predict(X_test_no_parking_constant)

forecasting
test_residuals = y_test - forecasting
RSS = sum((test_residuals)**2)   #Residual Sum of Squares

TSS = sum((y_test - y_test.mean())**2)   #Total Sum of Squares

rSquared = (TSS - RSS)/TSS

RMSE = np.sqrt(RSS/len(test_residuals))

ratio = RMSE/y_test.mean()



print("The Residual Sum of Squares: ", RSS)

print("The total sum of squares: ", TSS)

print("The R squared: ", rSquared)

print("The Root-mean squared error: ", RMSE)

print("The ratio between the RMSE and the mean of the test data (y_test.mean()): ", round(ratio,4))
X_test_no_parking_constant.loc[495,:]
guess = model.get_prediction(X_test_no_parking_constant.loc[495,:])
guess.summary_frame(alpha = 0.05)  #setting the value for alpha. Our confidence interval will be of 95%
np.exp(X_test_no_parking_constant.loc[495,:])-1   #just taking the row labeled as 495 as a sample.
print("Upper confidence interval",np.exp(7.880193)-1)

print("Expected value: ", np.exp(7.83814)-1)

print("Lower confidence interval: ", np.exp(7.796087)-1)
print("Real data point: ",np.exp(7.824446)-1)  #Very good!