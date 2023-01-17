# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Import the numpy and pandas package



import numpy as np

import pandas as pd



# Data Visualisation



import matplotlib.pyplot as plt 

import seaborn as sns
housing = pd.DataFrame(pd.read_csv("../input/Housing.csv"))
# Check the head of the dataset

housing.head()
housing.shape
housing.info()
housing.describe()
# Checking Null values

housing.isnull().sum()*100/housing.shape[0]

# There are no NULL values in the dataset, hence it is clean.
# Outlier Analysis

fig, axs = plt.subplots(2,3, figsize = (10,5))

plt1 = sns.boxplot(housing['price'], ax = axs[0,0])

plt2 = sns.boxplot(housing['area'], ax = axs[0,1])

plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])

plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])

plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])

plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])



plt.tight_layout()
# Outlier Treatment

# Price and area have considerable outliers.

# We can drop the outliers as we have sufficient data.
# outlier treatment for price

plt.boxplot(housing.price)

Q1 = housing.price.quantile(0.25)

Q3 = housing.price.quantile(0.75)

IQR = Q3 - Q1

housing = housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]
# outlier treatment for area

plt.boxplot(housing.area)

Q1 = housing.area.quantile(0.25)

Q3 = housing.area.quantile(0.75)

IQR = Q3 - Q1

housing = housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]
# Outlier Analysis

fig, axs = plt.subplots(2,3, figsize = (10,5))

plt1 = sns.boxplot(housing['price'], ax = axs[0,0])

plt2 = sns.boxplot(housing['area'], ax = axs[0,1])

plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])

plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])

plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])

plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])



plt.tight_layout()
sns.pairplot(housing)

plt.show()
plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.boxplot(x = 'mainroad', y = 'price', data = housing)

plt.subplot(2,3,2)

sns.boxplot(x = 'guestroom', y = 'price', data = housing)

plt.subplot(2,3,3)

sns.boxplot(x = 'basement', y = 'price', data = housing)

plt.subplot(2,3,4)

sns.boxplot(x = 'hotwaterheating', y = 'price', data = housing)

plt.subplot(2,3,5)

sns.boxplot(x = 'airconditioning', y = 'price', data = housing)

plt.subplot(2,3,6)

sns.boxplot(x = 'furnishingstatus', y = 'price', data = housing)

plt.show()
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = housing)

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
y_train = df_train.pop('price')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)
rfe = RFE(lm, 6)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
# Calculate the VIFs for the model

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lm.predict(X_train_rfe)
res = (y_train_price - y_train)
# Importing the required libraries for plots.

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
plt.scatter(y_train,res)

plt.show()
# There may be some relation in the error terms.
num_vars = ['area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
# Adding constant variable to test dataframe

X_test = sm.add_constant(X_test)
# Now let's use our model to make predictions.
# Creating X_test_new dataframe by dropping variables from X_test

X_test_rfe = X_test[X_train_rfe.columns]
# Making predictions

y_pred = lm.predict(X_test_rfe)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label