# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
car = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")
car.head()
car['current']= 2020
car['age']=car['current']-car['year']
car.drop(['current','year','name'],axis=1,inplace=True)

car.head()
car.info()
car.shape
car.describe()
# percentage of missing values in each column

round(100*(car.isnull().sum()/len(car)),2).sort_values(ascending = False)
# percentage of missing values in each row

round(100*(car.isnull().sum(axis=1)/len(car)),2).sort_values(ascending = False)
car_dub=car.copy()

# Checking for duplicates and dropping the entire duplicate row if any

car_dub.drop_duplicates(subset=None, inplace=True)
car_dub.shape
car.shape
car=car_dub

car.head()
car.info()
car.shape
for col in car:

    print(car[col].value_counts(ascending=False), '\n\n\n')
#To hold original data & column after duplicates are removed

car_o=car.copy()
car.info()
# Convert to 'category' data type

car['fuel']=car['fuel'].astype('category')

car['seller_type']=car['seller_type'].astype('category')

car['transmission']=car['transmission'].astype('category')

car['owner']=car['owner'].astype('category')
car.info()
# This code does 3 things:

# 1) Create Dummy variable

# 2) Drop original variable for which the dummy was created

# 3) Drop first dummy variable for each set of dummies created.



car = pd.get_dummies(car, drop_first=True)

car.info()
# Check the shape before spliting



car.shape

# Check the info before spliting



car.info()
from sklearn.model_selection import train_test_split



# We should specify 'random_state' so that the train and test data set always have the same rows, respectively



np.random.seed(0)

df_train, df_test = train_test_split(car, train_size = 0.70, test_size = 0.30, random_state = 100)
df_train.info()
df_train.shape
df_test.info()
df_test.shape
df_train.info()

df_train.columns
# Create a new dataframe of only numeric variables:



car_n=df_train[[ 'selling_price', 'km_driven', 'age']]



sns.pairplot(car_n, diag_kind='kde')

plt.show()
df_train.info()
# Build boxplot of all categorical variables (before creating dummies) againt the target variable 'selling_price' 

# to see how each of the predictor variable stackup against the target variable.



plt.figure(figsize=(25, 10))

plt.subplot(2,2,1)

sns.boxplot(x = 'fuel', y = 'selling_price', data = car_o)

plt.subplot(2,2,2)

sns.boxplot(x = 'seller_type', y = 'selling_price', data = car_o)

plt.subplot(2,2,3)

sns.boxplot(x = 'transmission', y = 'selling_price', data = car_o)

plt.subplot(2,2,4)

sns.boxplot(x = 'owner', y = 'selling_price', data = car_o)



plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated. Note:

# here we are considering only those variables (dataframe: car) that were chosen for analysis



plt.figure(figsize = (25,20))

sns.heatmap(car.corr(), annot = True, cmap="RdBu")

plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Checking the values before scaling

df_train.head()
df_train.columns
 #Apply scaler() to all the numeric variables



num_vars = ['selling_price', 'km_driven', 'age']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
y_train = df_train.pop('selling_price')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 7

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 7)             # running RFE

rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
import statsmodels.api as sm



# Add a constant

X_train_lm1 = sm.add_constant(X_train_rfe)



# Create a first fitted model

lr1 = sm.OLS(y_train, X_train_lm1).fit()
# Check the parameters obtained



lr1.params
# Print a summary of the linear regression model obtained

print(lr1.summary())
X_train_new = X_train_rfe.drop(["fuel_Electric"], axis = 1)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm2 = sm.add_constant(X_train_new)



# Create a first fitted model

lr2 = sm.OLS(y_train, X_train_lm2).fit()
# Check the parameters obtained



lr2.params
# Print a summary of the linear regression model obtained

print(lr2.summary())
X_train_new = X_train_new.drop(["owner_Test Drive Car"], axis = 1)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm3 = sm.add_constant(X_train_new)



# Create a first fitted model

lr3 = sm.OLS(y_train, X_train_lm3).fit()
# Check the parameters obtained



lr3.params
# Print a summary of the linear regression model obtained

print(lr3.summary())
X_train_new = X_train_new.drop(["seller_type_Trustmark Dealer"], axis = 1)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm4 = sm.add_constant(X_train_new)



# Create a first fitted model

lr4 = sm.OLS(y_train, X_train_lm4).fit()
# Check the parameters obtained



lr4.params
# Print a summary of the linear regression model obtained

print(lr4.summary())
y_train_pred = lr4.predict(X_train_lm4)
res = y_train-y_train_pred

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((res), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
car_n=car[[ 'selling_price', 'km_driven', 'age']]



sns.pairplot(car_n, diag_kind='kde')

plt.show()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif

#Apply scaler() to all the numeric variables



num_vars = ['selling_price', 'km_driven', 'age']



df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
df_test.head()
df_test.describe()
y_test = df_test.pop('selling_price')

X_test = df_test

X_test.info()

#Selecting the variables that were part of final model.

col1=X_train_new.columns

X_test=X_test[col1]

# Adding constant variable to test dataframe

X_test_lm4 = sm.add_constant(X_test)

X_test_lm4.info()
# Making predictions using the final model (lr6)



y_pred = lr4.predict(X_test_lm4)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
# We already have the value of R^2 (calculated in above step)



r2=0.3618371256083056 
# Get the shape of X_test

X_test.shape
# n is number of rows in X



n = X_test.shape[0]





# Number of features (predictors, p) is the shape along axis 1

p = X_test.shape[1]



# We find the Adjusted R-squared using the formula



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2