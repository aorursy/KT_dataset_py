#Importing necessary libraries



import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

pd.options.display.max_columns = 100

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

import os
#Reading the data set as a dataframe



cars = pd.read_csv('../input/geely-auto/CarPriceAssignment.csv')
#Viewing the cars dataframe



cars.head()
#Dimensions of the dataframe

cars.shape
#Checking the dataframe for any null values

cars.info()
#Getting a statistical view of the numerical variables of the dataframe

cars.describe()
# Dropping car_ID variable since it has nothing to do with price



cars.drop('car_ID', axis=1, inplace=True)
cars.head(10)
plt.figure(figsize=(10,10))

sns.pairplot(cars)

plt.show()
#Correlations of price with other numeric variables



cars[cars.columns[1:]].corr()['price'][:].round(2).sort_values(ascending=True)
cars.drop(columns = ['carheight','stroke','compressionratio','peakrpm'], axis=1, inplace=True)
cars['symboling'].value_counts()
#Categorising symboling values -3 to 0 as Low Risk and the remaining positive values as High Risk



def categorise(x):

    if(-3 <= x <= 0):

        return "Low Risk"

    else:

        return "High Risk"

        

cars['symboling'] = cars['symboling'].apply(lambda x: categorise(x))
cars.head()
#Extract the Company name from 'CarName' variable



cars['company'] = cars['CarName'].apply(lambda x: x.split(' ')[0])

cars['company'] = cars['company'].str.lower()
#Correcting the incorrect company names



def compName(x):

    if (x == "vw" or x == "vokswagen"):

        return "volkswagen"

    elif(x == "toyouta"):

        return "toyota"

    elif(x == "porcshce"):

        return "porsche"

    elif(x == "maxda"):

        return "mazda"

    

    else:

        return x

    

cars['company'] = cars['company'].apply(lambda x: compName(x))
#Dropping the CarName variable



cars.drop('CarName',axis=1,inplace=True)
cars.head()
plt.figure(figsize=(25,25))

fig_num = 0

def plot_categorical(var):       #Function to plot boxplots for all categorical variables

    plt.subplot(3,4, fig_num)

    sns.boxplot(x = var, y = 'price', data = cars)



categorical_vars = cars.dtypes[cars.dtypes==object].index

for var in categorical_vars:

    fig_num = fig_num + 1

    plot_categorical(var)



plt.show()
#Dropping doornumber variable from the dataset



cars.drop('doornumber', axis=1, inplace=True)
#Moving the price column to the front of the dataframe for better readability



cars = cars[['price','symboling','fueltype','aspiration','carbody','drivewheel','enginelocation','wheelbase','carlength','carwidth',

 'curbweight','enginetype','cylindernumber','enginesize','fuelsystem','boreratio','horsepower','company']]
cars.head()
def binary_map(x):

    cars[x] = cars[x].astype("category").cat.codes



binary_categorical_vars = ['symboling','fueltype','aspiration','enginelocation']

for var in binary_categorical_vars:

    binary_map(var)
cars.head()
print("Engine Type")

print(cars['enginetype'].value_counts(normalize=True).round(2))

print("\n")

print("Drivewheel")

print(cars['drivewheel'].value_counts(normalize=True).round(2))

print("\n")

print("Carbody")

print(cars['carbody'].value_counts(normalize=True).round(2))

print("\n")

print("Fuel System")

print(cars['fuelsystem'].value_counts(normalize=True).round(2))
def eng_map(x):

    if("ohc" in x):

        return 1

    else:

        return 0



cars['enginetype'] = cars.enginetype.apply(lambda x: eng_map(x))
cars['enginelocation'].value_counts(normalize=True).round(2)
cars.drop('enginelocation', axis = 1, inplace=True)
# Converting "cylindernumber" values to its corresponding number



cars['cylindernumber'].replace({"two":2,"three":3,"four":4,"five":5, "six":6,"eight":8,"twelve":12}, inplace=True)
#Get the dummy variables for carbody and store in separate variable "carbody_dummies"



carbody_dummies = pd.get_dummies(cars['carbody'],prefix='carbody',drop_first=True)

cars = pd.concat([cars,carbody_dummies], axis=1)

cars.drop('carbody',axis=1,inplace=True)
#Get the dummy variables for drivewheel and store in separate variable "drivewheel_dummies"



drivewheel_dummies = pd.get_dummies(cars['drivewheel'],prefix='dw',drop_first=True)

cars = pd.concat([cars,drivewheel_dummies], axis=1)

cars.drop('drivewheel',axis=1,inplace=True)
#Get the dummy variables for company and store in separate variable "company_dummies"



company_dummies = pd.get_dummies(cars['company'],prefix='comp',drop_first=True)

cars = pd.concat([cars,company_dummies], axis=1)

cars.drop('company',axis=1,inplace=True)
#Get the dummy variables for fuelsystem and store in separate variable "fuelsys_dummies"



fuelsys_dummies = pd.get_dummies(cars['fuelsystem'],prefix='dw',drop_first=True)

cars = pd.concat([cars,fuelsys_dummies], axis=1)

cars.drop('fuelsystem',axis=1,inplace=True)
cars.head()
cars.shape
cars.describe()
# from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train, df_test = train_test_split(cars, train_size = 0.7, test_size = 0.3, random_state=100)
# from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables



num_vars = ['price','wheelbase','carlength','carwidth','curbweight','cylindernumber','enginesize','boreratio','horsepower']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
y_train = df_train.pop('price')

X_train = df_train
#Creating an object of LinearRegression class and using RFE to get the top 20 variables from the dataset



lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm,20)

rfe = rfe.fit(X_train, y_train)
#Collecting the 20 variables selected by RFE



cols = X_train.columns[rfe.support_]

cols
#Selecting these 20 variables from X_train data set and assign to new variable X_train_rfe



X_train_rfe = X_train[cols]
#Add a constant to X_train_rfe data set using statsmodels.api library as sm



X_train_lm = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train, X_train_lm).fit()  # Running the linear model

print(lm.summary())                     # Viewing the summary of the linear model
#Calculating the VIF of the variables using variance_inflation_factor library



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_rfe.drop('curbweight',axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
# Calculate the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('comp_peugeot',axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('comp_isuzu',axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop(['enginesize'], axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('cylindernumber', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('boreratio', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('enginetype', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('comp_saab', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('comp_subaru', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('comp_audi', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('carwidth', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('carbody_sedan', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop('carbody_wagon', axis=1)
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train, X_train_lm).fit()

print(lm.summary())
#Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new.head()
X_train_lm.head()     # Training set with the constant
y_train_pred = lm.predict(X_train_lm)

error = y_train - y_train_pred
# Plot the histogram of the error terms



fig = plt.figure()

sns.distplot(error, bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18) 
plt.figure(figsize=(5,5))

sns.regplot(y_train_pred,error)

plt.xlabel('y_train_pred')

plt.ylabel('Error')
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables



num_vars_test = ['price','wheelbase','carlength','carwidth','curbweight','cylindernumber','enginesize','boreratio','horsepower']



df_test[num_vars_test] = scaler.transform(df_test[num_vars_test])
df_test.head()
y_test = df_test.pop('price')

X_test = df_test
# Creating X_test_new dataframe by only selecting variables present in the X_train training set



X_test_new = X_test[X_train_new.columns]



# Adding a constant variable

X_test_new = sm.add_constant(X_test_new)
y_test_pred = lm.predict(X_test_new)
fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_test_pred', fontsize=16)   
from sklearn.metrics import r2_score

r2_score(y_test, y_test_pred)