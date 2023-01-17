# warning

import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
# Importing Bike dataset

Bike = pd.read_csv('../input/bike-sharing-assignment/day.csv')
# Check the head of the dataset

Bike.head()
# chech the rows and column

Bike.shape
#check the information of dataset



Bike.info()
Bike.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# Renaming some variable

Bike.rename(columns={'yr':'year','mnth':'month','temp':'temperature','hum':'humidity','cnt':'count',},inplace=True)
Bike.head()
#checking if there is missing values in the dataset



import missingno as mn

mn.matrix(Bike)
#Dropping the unnecessary variable 

Bike.drop(['instant','dteday','casual','registered','holiday'],axis=1,inplace=True)
Bike.head()
Bike['season']= Bike['season'].map({1:"spring",2:"summer",3:"fall",4:"winter"})
sns.barplot('season','count',data=Bike)
sns.barplot(x='year',y='count',data=Bike)
Bike['month']= Bike['month'].map({1:"jan",2:"feb",3:"mar",4:"april",5:"may",6:"june",

                                7:"july",8:"aug",9:"sept",10:"oct",11:"nov",12:"dec"})
sns.barplot(x='month',y='count',data=Bike)
Bike['weekday']= Bike['weekday'].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
sns.barplot(x='weekday',y='count',data=Bike)
sns.barplot(x='workingday',y='count',data=Bike)
Bike['weathersit']= Bike['weathersit'].map({1:"Clear",2:"Mist",3:"Light Snow",4:"Heavy Rain"})

sns.barplot(x='weathersit',y='count',data=Bike)
sns.distplot(Bike['count'])
sns.pairplot(Bike)

plt.show()
plt.figure(figsize = (10,15))

sns.heatmap(Bike.corr(), annot = True, cmap="YlGnBu")

plt.show()
#dropping highly correlated variable

Bike.drop(['atemp'],axis=1,inplace=True)
Bike.head()
# Let's drop the first column from 'season','month','weekday','weathersit' Bike using 'drop_first = True'

seasons = pd.get_dummies(Bike['season'],drop_first=True)

month = pd.get_dummies(Bike['month'],drop_first=True)

weekday = pd.get_dummies(Bike['weekday'],drop_first=True)

weathersit = pd.get_dummies(Bike['weathersit'],drop_first=True)

working_day = pd.get_dummies(Bike['workingday'],drop_first=True)
# Add the results to the original Bike dataframe



Bike = pd.concat([Bike, seasons,month,weekday,weathersit,working_day], axis = 1)



Bike.head()
# Dropping  variables 

# As we already created dummy variable for 'season','month','weekday','weathersit' ,so drop those variable.



Bike.drop(['season','month','weekday','weathersit','workingday'], axis = 1, inplace = True)

Bike.head()
# We specify this so that the train and test data set always have the same rows, respectively

from sklearn.model_selection import train_test_split



np.random.seed(0)

Bike_train, Bike_test = train_test_split(Bike, train_size = 0.7, test_size = 0.3, random_state = 100)
#check the shape of train dataset 

print(Bike_train.shape)



#check the shape of train dataset

print(Bike_test.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['humidity','temperature','windspeed','count']



Bike_train[num_vars] = scaler.fit_transform(Bike_train[num_vars])
Bike_train.head()
Bike_train.describe()
y_train = Bike_train.pop('count')

X_train = Bike_train
y_train.head()
X_train.head()
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 10)             # running RFE

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

lm.summary()
X_train_drop = X_train_rfe.drop('Sun',1)

X_train_drop
X_train_2 = sm.add_constant(X_train_drop)



lm_1=sm.OLS(y_train,X_train_2).fit()



lm_1.summary()
X_train_drop1 = X_train_2.drop('const',1)

X_train_drop1
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_drop1

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_drop2 =X_train_drop1.drop('humidity',1)

X_train_drop2
X_train_3 = sm.add_constant(X_train_drop2)



lm_3= sm.OLS(y_train,X_train_3).fit()



lm_3.summary()
X_train_drop3 = X_train_3.drop('const',1)

X_train_drop3
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_drop3

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lm_3.predict(X_train_3)
fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
num_vars = ['humidity','temperature','windspeed','count']





Bike_test[num_vars] = scaler.transform(Bike_test[num_vars])
Bike_test.head()
Bike_test.describe()
y_test = Bike_test.pop('count')

X_test = Bike_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_drop3.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)



X_test_new.head()
# Making predictions

y_test_pred = lm_3.predict(X_test_new)
from sklearn.metrics import r2_score

r2_score(y_true= y_test , y_pred = y_test_pred )
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_test_pred', fontsize=16)                         # Y-label