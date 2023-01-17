# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



from scipy.stats import norm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cars = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

cars.info()
cars.head()
sns.distplot(cars.Selling_Price)
sns.distplot(np.log(cars.Selling_Price), fit=norm, kde=False)
sns.distplot(np.log1p(cars.Selling_Price), fit=norm, kde=False)
sns.distplot(np.sqrt(cars.Selling_Price), fit=norm, kde=False)
cars.Selling_Price = np.log(cars.Selling_Price)

cars['Present_Price'] = np.log(cars['Present_Price'])
cars2 = pd.read_csv("../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")

cars2.info()
cars2.head()
sns.distplot(cars2.selling_price)
sns.distplot(np.log(cars2.selling_price), fit=norm, kde=False)
cars2.selling_price = np.log(cars2.selling_price)
cars['Car_Name'].nunique()
cars2['name'].nunique()
cars['Car_Name'].value_counts()[:35]
cars2['name'].value_counts()[:35]
fig = plt.figure(figsize=(10, 20))



plt.title('Selling price of cars by car name')



price_order = cars.groupby('Car_Name')['Selling_Price'].mean().sort_values(ascending=False).index.values



sns.boxplot(data=cars, x='Selling_Price', y='Car_Name', 

            order=price_order)
fig = plt.figure(figsize=(10, 20))



plt.title('Selling price of cars by car name')



price_order = cars2.groupby('name')['selling_price'].mean().sort_values(ascending=False).index.values[:75]



sns.boxplot(data=cars2, x='selling_price', y='name', 

            order=price_order)
mean_price = cars.groupby('Car_Name')['Selling_Price'].mean().reset_index()

mean_price.sort_values(by='Selling_Price', ascending=False).head(5)
def plot_price(feature, title=''):

    fig = plt.figure(figsize=(8, 5))

    

    plt.title(title)



    price_order = cars.groupby(feature)['Selling_Price'].mean().sort_values(ascending=False).index.values



    sns.boxplot(data=cars, y='Selling_Price', x=feature, 

                order=price_order)

    



def plot_price2(feature, title='', width=8, height=5):

    fig = plt.figure(figsize=(width, height))

    

    plt.title(title)



    price_order = cars2.groupby(feature)['selling_price'].mean().sort_values(ascending=False).index.values



    sns.boxplot(data=cars2, y='selling_price', x=feature, 

                order=price_order)



plot_price('Fuel_Type', 'Selling price of cars by fuel type')
mean_price = cars.groupby('Fuel_Type')['Selling_Price'].mean().reset_index()

mean_price.sort_values(by='Selling_Price', ascending=False)
plot_price2('fuel', 'Selling price of cars by fuel type')
mean_price = cars2.groupby('fuel')['selling_price'].mean().reset_index()

mean_price.sort_values(by='selling_price', ascending=False)
plot_price('Seller_Type', 'Selling price of cars by seller type')
mean_price = cars.groupby('Seller_Type')['Selling_Price'].mean().reset_index()

mean_price.sort_values(by='Selling_Price', ascending=False)
plot_price2('seller_type', 'Selling price of cars by seller type')
mean_price = cars2.groupby('seller_type')['selling_price'].mean().reset_index()

mean_price.sort_values(by='selling_price', ascending=False)
plot_price('Year', 'Selling price of cars by year')
fig = plt.figure(figsize=(10, 5))

sns.distplot(cars[cars['Year']==2015].Selling_Price)
plot_price2('year', 'Selling price of cars by year', width=15)
plot_price('Owner', 'Selling price of cars by owner')
plot_price2('owner', 'Selling price of cars by owner')
plot_price('Transmission', 'Selling price of cars by transmission')
plot_price2('transmission', 'Selling price of cars by transmission')
cars2.info()
sns.lmplot(x='Kms_Driven', y='Selling_Price', data=cars)

ax = plt.gca()

ax.set_title('Selling price of cars by kms driven')
sns.distplot(cars['Kms_Driven'])
sns.distplot(np.sqrt(cars['Kms_Driven']), fit=norm, kde=False)
sns.distplot(np.log(cars['Kms_Driven']), fit=norm, kde=False)
cars['Kms_Driven'] = np.log(cars['Kms_Driven'])



sns.lmplot(x='Kms_Driven', y='Selling_Price', data=cars)

ax = plt.gca()

ax.set_title('Selling price of cars by kms driven')
sns.lmplot(x='km_driven', y='selling_price', data=cars2)

ax = plt.gca()

ax.set_title('Selling price of cars by kms driven')
sns.distplot(cars2['km_driven'])
cars2['km_driven'] = np.log(cars2['km_driven'])

sns.distplot(cars2['km_driven'])
sns.lmplot(x='km_driven', y='selling_price', data=cars2)

ax = plt.gca()

ax.set_title('Selling price of cars by kms driven')
sns.lmplot(x='Present_Price', y='Selling_Price', data=cars)

ax = plt.gca()

ax.set_title('Selling price of cars by present price')
cars = pd.concat([cars, pd.get_dummies(cars['Year'],prefix='Year')], axis=1)

cars.drop(['Year'],axis=1,inplace=True)
cars2 = pd.concat([cars2, pd.get_dummies(cars2['year'],prefix='Year')], axis=1)

cars2.drop(['year'],axis=1,inplace=True)
new_fuel = pd.get_dummies(cars['Fuel_Type'],prefix='Fuel')

new_seller = pd.get_dummies(cars['Seller_Type'],prefix='Seller')

new_transmission = pd.get_dummies(cars['Transmission'],prefix='Transmission')

new_owner = pd.get_dummies(cars['Owner'],prefix='Owner')



frames = [cars, new_fuel, new_seller, new_transmission, new_owner]

temp = pd.concat(frames, axis=1)

temp.drop(['Fuel_Type','Seller_Type','Transmission','Owner'],axis=1,inplace=True)

cars = temp

cars.head()
new_fuel = pd.get_dummies(cars2['fuel'],prefix='Fuel')

new_seller = pd.get_dummies(cars2['seller_type'],prefix='Seller')

new_transmission = pd.get_dummies(cars2['transmission'],prefix='Transmission')

# new_owner = pd.get_dummies(cars2['owner'],prefix='Owner')



# frames = [cars2, new_fuel, new_seller, new_transmission, new_owner]

frames = [cars2, new_fuel, new_seller, new_transmission]

temp = pd.concat(frames, axis=1)

temp.drop(['fuel','seller_type','transmission'],axis=1,inplace=True)

cars2 = temp

cars2.head()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(cars['Car_Name'])

cars['Car_Name'] = le.fit_transform(cars['Car_Name'])

cars.head()
cars.info()
le2 = LabelEncoder()

le2.fit(cars2['name'])

cars2['name'] = le.fit_transform(cars2['name'])

le2.fit(cars2['owner'])

cars2['owner'] = le.fit_transform(cars2['owner'])

cars2.head()
cars2.info()
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



import xgboost as xgb
y = cars.Selling_Price

X = cars.drop(columns='Selling_Price', axis=1)

X.head(10)
pd.DataFrame(y).head(10)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
def final_predictions(model, name):

    new_model = model.fit(X_train, y_train)

    pred = np.exp(model.predict(X_valid))

    print("============= %s and Shuffle Split =============" %name)

    print("Accuracy: %f" %(r2_score(np.exp(y_valid), pred)))

    print("MSE: %f" %(mean_squared_error(np.exp(y_valid), pred)))

    print("MAE: %f" %(mean_absolute_error(np.exp(y_valid), pred)))

    

    new_model = model.fit(X, y)

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

    cvs = cross_val_score(model, X, y, cv=cv)

    print('Shuffle and cross validate: %s \nAverage: %.2f' %(cvs, cvs.mean()))
lr_model = LinearRegression()

final_predictions(lr_model, 'Linear Regression')
rf_model = RandomForestRegressor(random_state=0)

final_predictions(rf_model, 'Random Forest Regressor')
dt_model = DecisionTreeRegressor(random_state=0)

final_predictions(dt_model, 'Decision Tree Regressor')
xgb_model = xgb.XGBRegressor()

final_predictions(xgb_model, 'XGBoost Regressor')
y = cars2.selling_price

X = cars2.drop(columns='selling_price', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

X.head(10)
pd.DataFrame(y).head(10)
lr_model = LinearRegression()

final_predictions(lr_model, 'Linear Regression')
rf_model = RandomForestRegressor(random_state=0)

final_predictions(rf_model, 'Random Forest Regressor')
dt_model = DecisionTreeRegressor(random_state=0)

final_predictions(dt_model, 'Decision Tree Regressor')
xgb_model = xgb.XGBRegressor()

final_predictions(xgb_model, 'XGBoost Regressor')