# data analysis and wrangling

import pandas as pd

import numpy as np



#sklearn

from sklearn import linear_model

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib notebook

%matplotlib inline
path = '../input/autos.csv'

cars = pd.read_csv(path, encoding='Latin1')
cars = cars.drop ('dateCrawled',1)

cars = cars.drop('offerType', 1) #only 6 cases of the other offer type, so will be droping that

cars = cars.drop('abtest', 1)

cars = cars.drop('dateCreated',1)

cars = cars.drop('postalCode',1)

cars = cars.drop('nrOfPictures',1)

#cars = cars.drop('postalCode',1)

cars = cars.drop('lastSeen',1)

cars = cars.drop('seller',1)
pd.to_numeric(cars.yearOfRegistration, errors='raise')

cars = cars[cars.yearOfRegistration > 2000]

cars = cars[cars.yearOfRegistration <= 2016]
cars = cars[cars.price > 100]

cars = cars[cars.price < 50000]
cars['vehicleType'] = cars['vehicleType'].fillna('missing') #filling missing values

cars['vehicleType'] = cars['vehicleType'].map({

                                                'missing':0,

                                                'limousine':1, 

                                                'kleinwagen':2,

                                                'kombi':3,

                                                'bus':4,

                                                'cabrio':5,

                                                'suv':6,

                                                'coupe':7,

                                                'andere':8})
cars['gearbox'] = cars['gearbox'].fillna('missing')

cars['gearbox'] = cars['gearbox'].map({

                                        'manuell':0,

                                        'automatik':1,

                                        'missing':2})
cars = cars [cars.powerPS > 50]

cars = cars [cars.powerPS < 1000]
cars.dropna(subset=['model'], inplace = True)

cars.reset_index(drop=True, inplace=True)

models = pd.DataFrame()

models['m'] = cars.model.unique()

keymodels = models.m

m=dict(keymodels)

m1 = { v : k for k, v in m.items()}

cars['model'] = cars['model'].map(m1)

inv_carsModelMap = {v: k for k, v in m.items()} #we will need that later to reverse mapping
cars['fuelType'] = cars['fuelType'].fillna('missing') #filling missing values

cars['fuelType'] = cars['fuelType'].map({

                                                'missing':0,

                                                'benzin':1, 

                                                'diesel':2,

                                                'lpg':3,

                                                'cng':4,

                                                'hybrid':5,

                                                'elektro':6,

                                                'andere':7})
cars['brand'] = cars['brand'].fillna('missing') #filling missing values

carsBrandMap = {'missing':0,'jeep':1, 'volkswagen':2, 'skoda':3,

'peugeot':4, 'ford':5, 'mazda':6, 'nissan':7,'renault':8, 'mercedes_benz':9,

'bmw':10, 'honda':11, 'fiat':12, 'mini':13, 'smart':14,'hyundai':15, 'audi':16,

'opel':17, 'volvo':18, 'mitsubishi':19, 'alfa_romeo':20,'kia':21, 'seat':22,

'lancia':23, 'subaru':24, 'citroen':25, 'chevrolet':26, 'dacia':27,'daihatsu':28,

'toyota':29, 'suzuki':30, 'chrysler':31, 'rover':32, 'porsche':33,'saab':34,

'daewoo':35, 'jaguar':36, 'land_rover':37, 'lada':38}

inv_carsBrandMap = {v: k for k, v in carsBrandMap.items()} #we will need that later to reverse mapping

cars['brand'] = cars['brand'].map(carsBrandMap)
cars.dropna(subset=['notRepairedDamage'], inplace = True)

cars['notRepairedDamage'] = cars['notRepairedDamage'].map({'ja':0, 'nein':1})
corr = cars.corr()

corr.loc[:,'price'].abs().sort_values(ascending=False)[1:]
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

a = sns.heatmap(corr, annot=True, fmt='.2f' ,cmap="YlGnBu")

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
def reg_analysis(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    #Calculate Variance score

    Variance_score = explained_variance_score(y_test, prediction)

    print ('Variance score : %.2f' %Variance_score)

    #Mean Absolute Error

    MAE = mean_absolute_error(y_test, prediction)

    print ('Mean Absolute Error : %.2f' %MAE)

    #Root Mean Squared Error

    RMSE = mean_squared_error(y_test, prediction)**0.5

    print ('Mean Squared Error : %.2f' %RMSE)

    #R² score, the coefficient of determination

    r2s = r2_score(y_test, prediction)

    print ('R2  score : %.2f' %r2s)
def sample_split(data):

    relevent_data = data[['price',

                      'vehicleType',

                      'yearOfRegistration',

                      'gearbox',

                      'powerPS',

                      'model',

                      'kilometer',

                      'fuelType',

                      'brand',

                      'notRepairedDamage']]

    relevent_cols = list(data)

    autos=relevent_data.values.astype(float)             

    Y = autos[:,0]

    X = autos[:,1:]

    test_size = .3

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = 3)

    return X_train, X_test, y_train, y_test
n_estimators = 100

model = RandomForestRegressor(n_estimators = n_estimators)

print ('Random Forest Regressor')

X_train, X_test, y_train, y_test = sample_split(cars)

reg_analysis(model,X_train, X_test, y_train, y_test) 
model = linear_model.LinearRegression()

print ('Linear Regression')

reg_analysis(model,X_train, X_test, y_train, y_test)
countModels = cars.groupby('model').size().nlargest(10).reset_index(name='top10')

countModels['model'] = countModels['model'].map(m)

order = list(countModels.model)
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(10,7))

a = sns.barplot(x=countModels.model, y=countModels.top10, data = countModels, order = order)
m1['golf']
#top5 = cars.loc[cars['model'].isin(d.index)]

top1 = cars.loc[cars['model'] == 1]
print (top1.size)

print (top1.size/cars.size)
n_estimators = 100

model = RandomForestRegressor(n_estimators = n_estimators)

print ('Random Forest Regressor')

X_train, X_test, y_train, y_test = sample_split(top1)

reg_analysis(model, X_train, X_test, y_train, y_test)
#model_selected = 1 #in case someone wants to focus on a specific model(s)

budget = 20000 #My budget for the car

km_per_year = 10000 #this is how many km I would do per year

use_for = 5 #years

old_cars_data = cars[['price',

                      'vehicleType',

                      'yearOfRegistration',

                      'gearbox',

                      'powerPS',

                      'model',

                      'kilometer',

                      'fuelType',

                      'brand',

                      'notRepairedDamage']]

#old_cars_data = old_cars_data.loc[old_cars_data['model'] == model_selected]

old_cars_data = old_cars_data.loc[old_cars_data['price'] <= budget]
pred_car_data = old_cars_data.copy()

pred_car_data['kilometer'] = pred_car_data['kilometer'] + (km_per_year*use_for)

pred_car_data['yearOfRegistration'] = pred_car_data['yearOfRegistration'] - use_for

autos = pred_car_data.values.astype(float)

X_pred = autos[:,1:]
n_estimators = 100

RFR = RandomForestRegressor(n_estimators = n_estimators)

X_train, X_test, y_train, y_test = sample_split(old_cars_data)

RFR.fit(X_train, y_train)

prices_for_pred_cars = RFR.predict(X_pred)

print (RFR.score)
old_cars_data['depreciation_price'] = prices_for_pred_cars

old_cars_data['depreciation_rate'] = (old_cars_data['price'] - old_cars_data['depreciation_price'])/use_for
old_cars_data.depreciation_rate.describe()
print ('percentage of cars having negative depreciacion rate %.2f %%' %((old_cars_data[old_cars_data["depreciation_rate"]< 0].count()/old_cars_data.count()).price*100))
topCarOptions = old_cars_data.sort_values('depreciation_rate', ascending=False).head(10)
topCarOptions['brand'] = topCarOptions['brand'].map(inv_carsBrandMap)

topCarOptions['model'] = topCarOptions['model'].map(m)

topCarOptions.head(100)