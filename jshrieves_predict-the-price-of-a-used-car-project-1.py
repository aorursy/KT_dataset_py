#Import relevant libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

set()

from sklearn.model_selection import train_test_split
#Import raw data 

raw_data = pd.read_csv(r'../input/1.04. Real-life example.csv')

raw_data.head()
#Explore the descriptive stats of numerical and qualitative variables

raw_data.describe(include='all')

#Eliminate unhelpful features

data = raw_data.drop(['Model'],axis=1)
#Determine missing values

data.isnull().sum()
#Since number of missing observations is less than 5% remove them

data_no_mv = data.dropna(axis=0)

data_no_mv.describe(include='all')
#Explore the probability distribution of each feature 

sns.distplot(data_no_mv['Price'])

sns.distplot(data_no_mv['Mileage'])
sns.distplot(data_no_mv['EngineV'])
sns.distplot(data_no_mv['Year'])
#The features price and milage have outliers. 

#The top 1% should be removed. 

q = data_no_mv['Price'].quantile(0.99)

data_2 = data_no_mv[data_no_mv['Price']<q]

q = data_2['Mileage'].quantile(0.99)

data_3 = data_2[data_2['Mileage']<q]
#EngineV has values above 6.5 which isn't possible. These values are erroneous and 

#will need to be removed. 

data_3 = data_2[data_2['EngineV']<6.5]



#Year has a few outliers probabley due to vintage cars. 

#The lower 1% should be removed. 

q = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year']>q]
#Data cleaning is complete. 

#Reset index, forget original index, double check descriptives

data_cleaned = data_4.reset_index(drop=True)

data_cleaned.describe(include='all')
#Check using scatterplots

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])

ax1.set_title('Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])

ax2.set_title('Price and EngineV')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])

ax3.set_title('Price and Mileage')

#Not linear

#Transform Price and add log values to data frame

log_price = np.log(data_cleaned['Price'])

data_cleaned['log_price'] = log_price
#Plot again using logs

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])

ax1.set_title('Log Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])

ax2.set_title('Log Price and EngineV')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])

ax3.set_title('Log Price and Mileage')
#Assume no endogeneity of regressors

#Assume normality, zero mean, and homoscedasticity 

#Assume no autocorrelation

#Check for multicollinearity. Logical that year and mileage would be correlated

from statsmodels.stats.outliers_influence import variance_inflation_factor 

variables = data_cleaned[['Mileage','EngineV','Year']]

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif["features"] = variables.columns

vif
#High multicollinearity for Year so we'll drop 

data_no_mc = data_cleaned.drop(['Year'],axis=1)
data_with_dummies = pd.get_dummies(data_no_mc, drop_first=True)

#Check data

data_with_dummies.head()
#Rearrange

data_with_dummies.columns.values
cols = ['log_price', 'Price', 'Mileage', 'EngineV', 'Brand_BMW',

       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',

       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',

       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',

       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

     
data_preprocessing = data_with_dummies[cols]
# The dependent variable is log price. 

targets = data_preprocessing['log_price']



# The inputs are everything but the dependent variable, so it can be dropped

inputs = data_preprocessing.drop(['log_price'],axis=1)
# Import the scaling module

from sklearn.preprocessing import StandardScaler



# Create a scaler object

scaler = StandardScaler()

# Fit the inputs 

scaler.fit(inputs)
# Scale features and store in a new variable 

inputs_scaled = scaler.transform(inputs)
# Import module for the split

from sklearn.model_selection import train_test_split



# Split variables with an 80-20 split 

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
# Create linear regression object

reg = LinearRegression()

# Fit the regression with the scaled TRAIN inputs and targets

reg.fit(x_train,y_train)
# Check the outputs of the regression

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)

# Name the axes

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

# X-axis and the y-axis should be the same

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
# Obtain the bias (intercept) of the regression

reg.intercept_

# Obtain the weights (coefficients) of the regression

reg.coef_
# Create a regression summary to compare with one-another

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
# Check the different categories in the 'Brand' variable to see which one

# is actually the benchmark

data_cleaned['Brand'].unique()



y_hat_test = reg.predict(x_test)

# Create scatter plot with the test targets and the test predictions

plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel('Targets (y_test)',size=18)

plt.ylabel('Predictions (y_hat_test)',size=18)

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
# Manually check these predictions by taking exponential of log_price

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

df_pf.head()
# Can also include the test targets in that data frame to manually compare

df_pf['Target'] = np.exp(y_test)

df_pf



# There's a lot of missing values so something is wrong with the data frame
# The old indexes are preserved

# Must reset the index and drop the old indexing

y_test = y_test.reset_index(drop=True)



# Check the result

y_test.head()
# Overwrite target column with the appropriate values

# by taking exponential of the test log price

df_pf['Target'] = np.exp(y_test)

df_pf
# Can also calculate the difference between targets and predictions

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Check how far off from the result percentage-wise by taking

# the absolute difference in % to order the data frame

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
# Explore the descriptives for additional insights

df_pf.describe()

pd.options.display.max_rows = 999

# Display result with only 2 digits after the dot 

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Sort by difference in % and manually check the model

df_pf.sort_values(by=['Difference%'])