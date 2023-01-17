# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pd.set_option('display.max_columns', 100)

# Import data set



cars = pd.read_csv('../input/usedcarscatalog/cars.csv')

cars.head()
cars = cars.drop(columns=['feature_0','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9'], axis=1)

cars.head()
cars.shape
cars.describe()
cars.columns
cars.dtypes
cars.isnull().sum()
cars = cars.dropna()

cars.isnull().sum()
# Calculate the age of the car



cars['age'] = 2020 - cars['year_produced']

cars.head()
# All numeric (float and int) variables in the dataset

cars_numeric = cars.select_dtypes(include=['float64', 'int64'])

cars_numeric.head()
# Correlation matrix

cor = cars_numeric.corr()

cor
# Figure size

plt.figure(figsize=(16,8))



# Heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
plt.figure(figsize=(25, 6))



plt.subplot(1,3,1)

plt1 = cars.manufacturer_name.value_counts().plot(kind='bar')

plt.title('Companies Histogram')

plt1.set(xlabel = 'Car company', ylabel='Frequency of company')



plt.subplot(1,3,2)

plt1 = cars.body_type.value_counts().plot(kind='bar')

plt.title('Body Type')

plt1.set(xlabel = 'Body Type', ylabel='Frequency of Body Type')



plt.subplot(1,3,3)

plt1 = cars.engine_type.value_counts().plot(kind='bar')

plt.title('Engine Type Histogram')

plt1.set(xlabel = 'Engine Type', ylabel='Frequency of Engine type')



plt.show()
plt.figure(figsize=(30, 10))



df = pd.DataFrame(cars.groupby(['manufacturer_name'])['price_usd'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Company Name vs Average Price')

plt.show()



df = pd.DataFrame(cars.groupby(['engine_fuel'])['price_usd'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Fuel Type vs Average Price')

plt.show()



df = pd.DataFrame(cars.groupby(['body_type'])['price_usd'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Car Type vs Average Price')

plt.show()
cars['price_usd'] = cars['price_usd'].astype('float64')

temp = cars.copy()



table = temp.groupby(['manufacturer_name'])['price_usd'].mean()

temp = temp.merge(table.reset_index(), how='left', on='manufacturer_name')

bins = [0,10000,25000,50000]

cars_bins = ['Budget','Medium', 'Highend']
temp.head()
cars['CarRange'] = pd.cut(temp['price_usd_y'], bins,  right=False, labels=cars_bins)

cars.head()
## We will leave out variables like "manufacturer_name","model_name","location region" 

## We will be using CarsRange variable instead of these as discussed above.
cars_new = cars[['transmission','color','odometer_value','engine_fuel','engine_has_gas','engine_type','engine_capacity','body_type'

                , 'has_warranty','state','drivetrain','is_exchangeable','number_of_photos', 'up_counter','duration_listed', 'age','CarRange','price_usd']]

cars_new.head()
# Define a function to generate dummy variables and merging it with data frame



def dummies(x,df):

    temp = pd.get_dummies(df[[x]], drop_first=True)

    df = pd.concat([df,temp], axis=1)

    df.drop([x], axis=1, inplace=True)

    return df



# Apply function to the cars_new df

cars_new = dummies('transmission', cars_new)

cars_new = dummies('color', cars_new)

cars_new = dummies('engine_fuel', cars_new)

cars_new = dummies('engine_has_gas', cars_new)

cars_new = dummies('engine_type', cars_new)

cars_new = dummies('body_type', cars_new)

cars_new = dummies('has_warranty', cars_new)

cars_new = dummies('state', cars_new)

cars_new = dummies('drivetrain', cars_new)

cars_new = dummies('is_exchangeable', cars_new)

cars_new = dummies('CarRange', cars_new)
cars_new.head()
cars_new.shape
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(cars_new, train_size=0.7, random_state=42) 
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_vars= ['odometer_value', 'engine_capacity', 'number_of_photos','up_counter','duration_listed', 'age','price_usd']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
# correlation



plt.figure(figsize = (30, 25))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# dividing variables in to X and y

y_train = df_train.pop('price_usd')

X_train = df_train
lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
# Building a model



def build_Lr_model(X,y):

    X = sm.add_constant(X) #add constant

    lm = sm.OLS(y,X).fit() #fit the model

    print(lm.summary())

    return X



def checkingVIF(X):

    vif = pd.DataFrame()

    vif['features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_1 = build_Lr_model(X_train_rfe, y_train)
checkingVIF(X_train_1)
lm = sm.OLS(y_train,X_train_1).fit()

y_train_price = lm.predict(X_train_1)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
num_vars= ['odometer_value', 'engine_capacity', 'number_of_photos','up_counter','duration_listed', 'age','price_usd']

df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
#Dividing into X and y

y_test = df_test.pop('price_usd')

X_test = df_test
# Now let's use our model to make predictions.

X_train_1 = X_train_1.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_train_1 = X_test[X_train_1.columns]



# Adding a constant variable 

X_train_1 = sm.add_constant(X_train_1)
# Making predictions

y_pred = lm.predict(X_train_1)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error
polynomial = PolynomialFeatures(degree=2)

polynomial_model = polynomial.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(polynomial_model,y, train_size=0.7, random_state=42)



# Build second LR model using polynomial features

lr_model_2 = LinearRegression().fit(X_train,y_train)





#PRedict the values

y_train_pred = lr_model_2.predict(X_train)



#Predict test values

y_test_pred = lr_model_2.predict(X_test)
print(lr_model_2.score(X_test, y_test))
#Dividing into X and y

y = cars_new.pop('price_usd')

X = cars_new
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=42)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



rf = RandomForestRegressor()



param_grid = { "criterion" : ["mse"]

              , "min_samples_leaf" : [1,5,1]

              , "min_samples_split" : [1,5,1]

              , "max_depth": [10]

              , "n_estimators": [500]}



gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)

print(gs.best_params_)
bp = gs.best_params_

forest = RandomForestRegressor(criterion=bp['criterion'],

                              min_samples_leaf=bp['min_samples_leaf'],

                              min_samples_split=bp['min_samples_split'],

                              max_depth=bp['max_depth'],

                              n_estimators=bp['n_estimators'])

forest.fit(X_train, y_train)



print('Score: %.2f' % forest.score(X_test, y_test))
important_features = pd.Series(data=forest.feature_importances_,index=X_train.columns)

important_features.sort_values(ascending=False,inplace=True)

important_features