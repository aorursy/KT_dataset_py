# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
housing = pd.read_csv(r'/kaggle/input/housing-simple-regression/Housing.csv')
# Check the head of the dataset

housing.head()
housing.shape
housing.info()
housing.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(housing)

plt.show()
plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.violinplot(x = 'mainroad', y = 'price', data = housing)

plt.subplot(2,3,2)

sns.violinplot(x = 'guestroom', y = 'price', data = housing)

plt.subplot(2,3,3)

sns.violinplot(x = 'basement', y = 'price', data = housing)

plt.subplot(2,3,4)

sns.violinplot(x = 'hotwaterheating', y = 'price', data = housing)

plt.subplot(2,3,5)

sns.violinplot(x = 'airconditioning', y = 'price', data = housing)

plt.subplot(2,3,6)

sns.violinplot(x = 'furnishingstatus', y = 'price', data = housing)

plt.show()
plt.figure(figsize = (10, 5))

sns.violinplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = housing)

plt.show()
# List of variables to map



varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})



# Applying the function to the housing list

housing[varlist] = housing[varlist].apply(binary_map)
# Check the housing dataframe now



housing.head()
# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'

status = pd.get_dummies(housing['furnishingstatus'])
# Check what the dataset 'status' looks like

status.head()
# Let's drop the first column from status df using 'drop_first = True'



status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)
# Add the results to the original housing dataframe



housing = pd.concat([housing, status], axis = 1)
# Now let's see the head of our dataframe.



housing.head()
# Drop 'furnishingstatus' as we have created the dummies for it



housing.drop(['furnishingstatus'], axis = 1, inplace = True)
housing.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (16, 10))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
plt.figure(figsize=[6,6])

plt.scatter(df_train.area, df_train.price)

plt.show()
y_train = df_train.pop('price')

X_train = df_train
# Check all the columns of the dataframe



housing.columns
#Build a linear model



import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train)



lr_1 = sm.OLS(y_train, X_train_lm).fit()



lr_1.params
print(lr_1.summary())
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables



X = X_train.drop('semi-furnished', 1,)
# Build a third fitted model

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model

print(lr_2.summary())
# Calculate the VIFs again for the new model



vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping highly correlated variables and insignificant variables

X = X.drop('bedrooms', 1)
# Build a second fitted model

X_train_lm = sm.add_constant(X)



lr_3 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model



print(lr_3.summary())
# Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X = X.drop('basement', 1)
# Build a fourth fitted model

X_train_lm = sm.add_constant(X)



lr_4 = sm.OLS(y_train, X_train_lm).fit()
print(lr_4.summary())
# Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lr_4.predict(X_train_lm)
# Plot the histogram of the error terms

fig = plt.figure()

plt.grid()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label

plt.show()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']



df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()
y_test = df_test.pop('price')

X_test = df_test
# Adding constant variable to test dataframe

X_test_m4 = sm.add_constant(X_test)
# Creating X_test_m4 dataframe by dropping variables from X_test_m4



X_test_m4 = X_test_m4.drop(["bedrooms", "semi-furnished", "basement"], axis = 1)
# Making predictions using the fourth model



y_pred_m4 = lr_4.predict(X_test_m4)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.grid()

plt.scatter(y_test, y_pred_m4)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)   

plt.show()
from sklearn.metrics import r2_score
r2_score_lr_train=0.676

print("R-squared Train:",r2_score_lr_train)
r2_score_lr_test=round(r2_score(y_test, y_pred_m4),3)

print("R-squared Test:",r2_score_lr_test)
lr_4.params.sort_values(ascending = False) 
