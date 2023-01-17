import numpy as np

import pandas as pd
car = pd.read_csv('../input/CarPriceAssignment.csv')

car.head()
print(car.isnull().any(axis=1).sum())

print(car.isnull().any(axis =0).sum())
car.CarName
new = car["CarName"].str.split(" ", n = 1, expand = True) 

  

# making separate first name column from new data frame 

car["carCompany"]= new[0] 

  

# Dropping old Name columns 

car.drop(columns =["CarName"], inplace = True) 
car.info()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(15, 12))

plt.subplot(4,3,1)

sns.boxplot(x = 'symboling', y = 'price', data = car)

plt.subplot(4,3,2)

sns.boxplot(x = 'carCompany', y = 'price', data = car)

plt.subplot(4,3,3)

sns.boxplot(x = 'fueltype', y = 'price', data = car)

plt.subplot(4,3,4)

sns.boxplot(x = 'aspiration', y = 'price', data = car)

plt.subplot(4,3,5)

sns.boxplot(x = 'doornumber', y = 'price', data = car)

plt.subplot(4,3,6)

sns.boxplot(x = 'carbody', y = 'price', data = car)

plt.subplot(4,3,7)

sns.boxplot(x = 'drivewheel', y = 'price', data = car)

plt.subplot(4,3,8)

sns.boxplot(x = 'enginelocation', y = 'price', data = car)

plt.subplot(4,3,9)

sns.boxplot(x = 'enginetype', y = 'price', data = car)

plt.subplot(4,3,10)

sns.boxplot(x = 'cylindernumber', y = 'price', data = car)

plt.subplot(4,3,11)

sns.boxplot(x = 'fuelsystem', y = 'price', data = car)

plt.show()
plt.figure(figsize = (20,10))  

sns.heatmap(car.corr(),annot = True, cmap="YlGnBu")
# NOt including symboling for creation of dummy variable as it is already in numerical ordered state



cat_col = ['carCompany','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']

car.head(10)
# To see the distribution of values in the various categorical columns

for col in car[cat_col]:

    print(car[col].value_counts())
car.drop(['enginelocation'], axis = 1, inplace = True)
cat_col = ['carCompany','fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem']
for col in car[cat_col]:

    print(car[col].unique())

    

# We see that few of the car company names are spelt incorrectly
car['carCompany'] = car['carCompany'].replace(['vokswagen', 'vw','maxda','Nissan','porcshce','toyouta'], ['volkswagen','volkswagen','mazda','nissan','porsche','toyota'])
for col in car[cat_col]:

    print(car[col].unique())
dummies = pd.get_dummies(car[cat_col])

dummies.head()
dummies.info()
for col in dummies:

    print(dummies[col].unique())
non_scalar = list(dummies.columns)

non_scalar
car = pd.concat([car, dummies], axis = 1)

car.info()
#Deleting the columns for which dummy variables are created

car.drop(['carCompany','fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem'], axis = 1, inplace = True)
car.drop('car_ID', axis =1, inplace = True)

car.info()
#from scipy import stats

#car = car[(np.abs(stats.zscore(car)) < 3).all(axis=1)]
car.shape
import pandas_profiling

pandas_profiling.ProfileReport(car)
# Dropping columns which the profiling is suggesting based on the correlation between the predictors

drop_col = ['cylindernumber_two','enginetype_l','fuelsystem_1bbl','fuelsystem_idi','fueltype_diesel','highwaympg']

car.drop(columns =drop_col, inplace = True) 
car.head()
from sklearn.model_selection import train_test_split



np.random.seed(0)

car_train, car_test = train_test_split(car, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols = [col for col in car.columns if col not in dummies.columns]

car[cols].head()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

car_train[cols] = scaler.fit_transform(car_train[cols])
car_train[cols].head()
car_train.describe()
plt.figure(figsize = (16, 10))

sns.heatmap(car_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
car_train.info()
# Finding the top correlations from the car_train dataframe

corr_matrix = car_train.corr().abs()

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

print(sol.head(40))
y_train = car_train.pop('price')

X_train = car_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 20)             # running RFE

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
print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_rfe = X_train_rfe.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_1 = X_train_rfe.drop('wheelbase', 1)

X_train_1 = sm.add_constant(X_train_1)
lm = sm.OLS(y_train,X_train_1).fit()   # Running the linear model
print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_1 = X_train_1.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_1

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_2 = X_train_1.drop('enginesize', 1)
X_train_2 = sm.add_constant(X_train_2)

lm = sm.OLS(y_train,X_train_2).fit()   # Running the linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_2 = X_train_2.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_2

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_3 = X_train_2.drop('cylindernumber_six', 1)

X_train_3 = sm.add_constant(X_train_3)

lm = sm.OLS(y_train,X_train_3).fit()   # Running the linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_3 = X_train_3.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_3

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_4 = X_train_3.drop('carwidth', 1)

X_train_4 = sm.add_constant(X_train_4)

lm = sm.OLS(y_train,X_train_4).fit()   # Running the linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_4 = X_train_4.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_4

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_5 = X_train_4.drop('cylindernumber_three', 1)

X_train_5 = sm.add_constant(X_train_5)

lm = sm.OLS(y_train,X_train_5).fit()   # Running the linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_5 = X_train_5.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_5

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_6 = X_train_5.drop('carCompany_saab', 1)

X_train_6 = sm.add_constant(X_train_6)

lm = sm.OLS(y_train,X_train_6).fit()   # Running the linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_6 = X_train_6.drop(['const'], axis=1)

vif = pd.DataFrame()

X = X_train_6

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_7 = X_train_6.drop('boreratio', 1)

X_train_7 = sm.add_constant(X_train_7)

lm = sm.OLS(y_train,X_train_7).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_7 = X_train_7.drop(['const'], axis=1)

X = X_train_7

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_8 = X_train_7.drop('enginetype_rotor', 1)

X_train_8 = sm.add_constant(X_train_8)

lm = sm.OLS(y_train,X_train_8).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_8 = X_train_8.drop(['const'], axis=1)

X = X_train_8

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_9 = X_train_8.drop('carCompany_volvo', 1)

X_train_9 = sm.add_constant(X_train_9)

lm = sm.OLS(y_train,X_train_9).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_9 = X_train_9.drop(['const'], axis=1)

X = X_train_9

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_10 = X_train_9.drop('enginetype_ohcf', 1)

X_train_10 = sm.add_constant(X_train_10)

lm = sm.OLS(y_train,X_train_10).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_10 = X_train_10.drop(['const'], axis=1)

X = X_train_10

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_11 = X_train_10.drop('carCompany_subaru', 1)

X_train_11 = sm.add_constant(X_train_11)

lm = sm.OLS(y_train,X_train_11).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_11 = X_train_11.drop(['const'], axis=1)

X = X_train_11

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_12 = X_train_11.drop('stroke', 1)

X_train_12 = sm.add_constant(X_train_12)

lm = sm.OLS(y_train,X_train_12).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_12 = X_train_12.drop(['const'], axis=1)

X = X_train_12

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_13 = X_train_12.drop('citympg', 1)

X_train_13 = sm.add_constant(X_train_13)

lm = sm.OLS(y_train,X_train_13).fit()   # Running the linear model

print(lm.summary())
vif = pd.DataFrame()

X_train_test = X_train_13.drop(['const'], axis=1)

X = X_train_test

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lm.predict(X_train_13)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   
car_test[cols] = scaler.transform(car_test[cols])
car_test.info()
lm.params
car_test_lm = car_test[['curbweight','peakrpm','carCompany_bmw','carCompany_porsche','carbody_convertible','cylindernumber_eight','cylindernumber_twelve']]
y_test = car_test.pop('price')
X_test = car_test_lm
X_test_lm = sm.add_constant(X_test)
y_pred_lm = lm.predict(X_test_lm)
X_test_lm.shape
from sklearn.metrics import r2_score

r2_score(y_test, y_pred_lm)