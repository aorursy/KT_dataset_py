# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing numpy and pandas libraries and also ignoring the warning if we may get

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd
cars = pd.read_csv("/kaggle/input/geely-auto/CarPriceAssignment.csv")  #Reading data from the csv file provided
#Check the head of the data

pd.set_option('display.max_columns', 500) #So that we can see all columns

cars.head()
#Car name includes company and model name. We are interested in company name only.

carName = cars.CarName.str.split(" ", n = 1, expand = True)

cars['CarCompany'] = carName[0]

cars.drop('CarName', axis = 1, inplace=True)

cars.CarCompany.unique()
cars['CarCompany'] = cars.CarCompany.str.upper()

cars['CarCompany'] = cars['CarCompany'].replace({'VW':'VOLKSWAGEN','VOKSWAGEN':'VOLKSWAGEN','TOYOUTA':'TOYOTA','PORCSHCE':'PORSCHE','MAXDA':'MAZDA'})

cars.CarCompany.unique()
#Getting the data type of each column

cars.info()
#Getting shape of the data

cars.shape
#Removing duplicates if any

cars = cars.drop_duplicates(keep=False)
#Getting the description of the data

cars.describe()
#Importing apropriate libraries

import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(cars)

plt.show()
#Heat map for the data

plt.figure(figsize=(12,10))

sns.heatmap(cars.corr(), annot=True)
#Since the pair plot and heatmap are very conjusted, let's try to get the correlation of each variable using .corr funtion

cars.corr()
#From the data dictionary we can get the categorical varibles. Let's plot boxplot for a few of them

plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)

sns.boxplot(x='fueltype',y='price',data = cars)

plt.subplot(3,3,2)

sns.boxplot(x='aspiration',y='price',data = cars)

plt.subplot(3,3,3)

sns.boxplot(x='doornumber',y='price',data = cars)

plt.subplot(3,3,4)

sns.boxplot(x='carbody',y='price',data = cars)

plt.subplot(3,3,5)

sns.boxplot(x='drivewheel',y='price',data = cars)

plt.subplot(3,3,6)

sns.boxplot(x='enginelocation',y='price',data = cars)

plt.subplot(3,3,7)

sns.boxplot(x='enginetype',y='price',data = cars)

plt.subplot(3,3,8)

sns.boxplot(x='cylindernumber',y='price',data = cars)

plt.subplot(3,3,9)

sns.boxplot(x='fuelsystem',y='price',data = cars)

plt.show()
plt.figure(figsize=(16,9))

sns.boxplot(x='CarCompany',y='price',data = cars)

plt.xticks(rotation=90);
cars.head()
# To get the list which are not of integer type

cars.select_dtypes(exclude=['int64', 'float64']).columns
# Let change the doornumber from string to integer values

cars['doornumber'] = cars['doornumber'].apply(lambda x: 2 if x=='two' else (4 if x=='four' else 0))
# Now we will check for if the car is diesel or not

cars.fueltype = cars.fueltype.apply(lambda x: 0 if x=='gas' else 1)

cars['DieselCar'] = cars.fueltype

cars = cars.drop('fueltype', axis = 1)

cars.head()
# Let map the car has the engine in front side to 1 else 0

cars.enginelocation = cars.enginelocation.apply(lambda x: 0 if x=='rear' else 1)

cars['FrontEngine'] = cars.enginelocation

cars = cars.drop('enginelocation', axis = 1)

cars.head()
# We'll map 1 for the TURBO cars



cars.aspiration = cars.aspiration.apply(lambda x: 0 if x=='std' else 1)

cars['Turbo'] = cars.aspiration

cars = cars.drop('aspiration', axis = 1)

cars.head()
# For engine type

cars.enginetype.value_counts()
cars['OHCEngine'] = cars['enginetype'].apply(lambda x: 1 if 'ohc' in x else 0)

cars = cars.drop('enginetype', axis = 1)

cars.head()
def risk_factor(x):

    m = {-3:0,-2:0,-1:1,0:2,1:3,2:4,3:5} # 5 means most risky car

    return m.get(x)



cars['RiskFactor'] = cars.symboling.apply(risk_factor)

cars = cars.drop('symboling', axis = 1)

cars.head()
def get_cyl_num(x):

    m = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12}

    return m.get(x)



cars.cylindernumber = cars.cylindernumber.apply(get_cyl_num)

cars.head()
status = pd.get_dummies(cars['drivewheel'], drop_first=True)  #It will create dummy variables.

status.head()
#Concating status and cars dataframe

cars = pd.concat([cars,status], axis = 1)

cars = cars.drop('drivewheel',axis = 1)   #Dropping `drivewheel` as it not required

cars.head()
#Let's deal with carbody now.

# We can categories each carbody using category codes



#First let's check the current carbody values

print(cars.carbody.unique())
#Now we will convert each value to a code

cars.carbody = cars.carbody.astype('category')

cars.carbody = cars.carbody.cat.codes

print(cars.carbody.unique())
cars.head()
#Current fuelsystem values are

print(cars.fuelsystem.unique())
#Let encode the values

cars.fuelsystem = cars.fuelsystem.astype('category')

cars.fuelsystem = cars.fuelsystem.cat.codes

print(cars.fuelsystem.unique())
cars.head()
#The way we created the dummy variables for the previous columns, we will follow the same procedure

Car_Name_df = pd.get_dummies(cars.CarCompany)

Car_Name_df.head()
#Let's check the counts for each brand

cars.CarCompany.value_counts()
#We can see, MERCURY car has only one value, we can delete it from Car_Name_df dataframe

Car_Name_df = Car_Name_df.drop('MERCURY', axis = 1)

#Then we can concate it with the main dataset i.e. cars

cars = pd.concat([cars,Car_Name_df], axis = 1)

cars.drop('CarCompany', axis=1, inplace=True)

cars.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

df_train, df_test = train_test_split(cars, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler #Importing library

scaler = MinMaxScaler()   #Getting MinMaxScaler
num_vars = [ 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight','enginesize','boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg','price']

#In num_vars we have columns that we need to scale.

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_test.head()
df_train.describe()
y_train = df_train.pop('price')   #Dependent Variable

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 10)             # running RFE

rfe = rfe.fit(X_train, y_train)
# Checking which columns were selected by the RFE

col = X_train.columns[rfe.support_]

print(col)
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
#Since statsmodels don't consider the constants or intercept by default, we will add constant

import statsmodels.api as sm  

X_train_rfe_lm = sm.add_constant(X_train_rfe)
lm1 = sm.OLS(y_train,X_train_rfe_lm).fit()    #Running Linear Model and filling the train data
print(lm1.summary())
#VIF Calculation for Model #1

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending= False)

print(vif)
X_train_rfe = X_train_rfe.drop('carwidth',axis=1)

X_train_rfe_lm = sm.add_constant(X_train_rfe)

lm2 = sm.OLS(y_train, X_train_rfe_lm).fit()

print(lm2.summary())

vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending= False)

print(vif)
X_train_rfe = X_train_rfe.drop('enginesize',axis=1)

X_train_rfe_lm = sm.add_constant(X_train_rfe)

lm3 = sm.OLS(y_train, X_train_rfe_lm).fit()

print(lm3.summary())

vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending= False)

print(vif)
X_train_rfe = X_train_rfe.drop('FrontEngine',axis=1)

X_train_rfe_lm = sm.add_constant(X_train_rfe)

lm4 = sm.OLS(y_train, X_train_rfe_lm).fit()

print(lm4.summary())

vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending= False)

print(vif)
final_model = lm4
y_train_price = lm4.predict(X_train_rfe_lm)   #getting the predicted values
# Importing the required libraries for plots.

import matplotlib.pyplot as plt

import seaborn as sns

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
num_vars = [ 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight','enginesize','boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg','price']

df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_rfe.columns]



# Adding a constant variable 

X_test_new_lm = sm.add_constant(X_test_new)
# Making predictions

y_pred = final_model.predict(X_test_new_lm)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label

from sklearn.metrics import r2_score, mean_squared_error

print("R-Square for test:")

print(round(r2_score(y_test, y_pred),4))
print("Mean Sqaure Error for test:")

print(round(mean_squared_error(y_test, y_pred),4))
final_model.params.sort_values()