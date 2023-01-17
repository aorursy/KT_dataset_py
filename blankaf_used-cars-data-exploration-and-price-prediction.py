import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
prep_cars=pd.read_csv('../input/autos.csv', encoding='latin_1')

prep_cars.columns
columns_to_keep = ['brand','model','vehicleType','yearOfRegistration',

                   'monthOfRegistration','kilometer','powerPS','fuelType',

                   'gearbox','abtest','notRepairedDamage','price']

prep_cars = prep_cars[columns_to_keep]

prep_cars.head(2)
print(len(prep_cars))
for col in prep_cars.columns:

    if col[:]=='vehicleType':

        prep_cars.rename(columns={col:'car_type'}, inplace=True)

    if col[:]=='yearOfRegistration': 

        prep_cars.rename(columns={col:'reg_year'}, inplace=True)

    if col[:]=='monthOfRegistration': 

        prep_cars.rename(columns={col:'reg_month'}, inplace=True)

    if col[:]=='notRepairedDamage':

        prep_cars.rename(columns={col:'damage'}, inplace=True)

        

prep_cars['damage'].fillna('norep', inplace=True)

prep_cars = prep_cars.dropna()

prep_cars.head()
                                     # Preparing for introduction of a new feature age 

new_cars4 = prep_cars[(prep_cars['reg_year']==2016) & (prep_cars['reg_month']==4)]

new_cars3 = prep_cars[(prep_cars['reg_year']==2016) & (prep_cars['reg_month']==3)]

new_cars2 = prep_cars[(prep_cars['reg_year']==2016) & (prep_cars['reg_month']==2)]

new_cars1 = prep_cars[(prep_cars['reg_year']==2016) & (prep_cars['reg_month']==1)]

print('Number of listed new cars registered in April 2016 is: ', new_cars4['brand'].count())

print(' in March 2016 it is: ', new_cars3['brand'].count())

print(' in February 2016 it is: ', new_cars2['brand'].count())

print(' and in January 2016 it is: ', new_cars1['brand'].count())      
new_cars0 = prep_cars[(prep_cars['reg_year']>2016)]

print('Number of listed cars with invalid registration year: ', new_cars0['brand'].count()) 
def age_count(yr, mt):                    # counting the age of the car as the time from

    return (2015-yr)*12+(12-mt)+3        # registration to April,1,2016

                                        # introducing a new attribute = car age in months  

prep_cars['age_months'] = age_count(prep_cars['reg_year'],prep_cars['reg_month'])        
prep_cars['km/1000'] = prep_cars['kilometer']/1000       # scaled feature in a new column

prep_cars.head(2)
columns_to_keep = ['brand','model','car_type','reg_year','age_months','km/1000',

                   'powerPS','fuelType','gearbox','abtest','damage','price']

prep_cars = prep_cars[columns_to_keep]

prep_cars.head(2)
sel_cars = prep_cars[(prep_cars.age_months >= 0)&(prep_cars.price <= 150000)&

                     (prep_cars.price > 100)]

sel_cars.head()                   # removing records with invalid or very extreme values
print('Number of cars after preprocessing is: ',len(sel_cars)) 
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

plt.title('BRANDS DISTRIBUTION')

g = sns.countplot(sel_cars['brand'])

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=90)
skoda_cars = sel_cars[sel_cars.brand =='skoda']

print('Number of Skoda cars on sale is:',len(skoda_cars),'and the mean price is',

       int(np.mean(skoda_cars.price)), 'Euros.')

audi_cars = sel_cars[sel_cars.brand =='audi']

print('Number of Audi cars on sale is:',len(audi_cars),'and the mean price is',

       int(np.mean(audi_cars.price)), 'Euros.')   
sns.set_context("notebook",font_scale=1.3)

plt.figure(figsize=(10,4))

plt.title('MODELS DISTRIBUTION FOR SKODA BRAND')

sns.countplot(skoda_cars['model'])
skoda_cars = skoda_cars[(skoda_cars.price < 50000)]      # removing a few apparent outliers

skoda_cars = skoda_cars[(skoda_cars.age_months < 450)]    # small additional cleaning

sns.set_context("notebook", font_scale=1.0)

sns.pairplot(skoda_cars,x_vars=['age_months'],y_vars='price',size=7)    
sns.set_context("notebook",font_scale=1.2)

plt.figure(figsize=(13,5))

plt.title('MODELS DISTRIBUTION FOR AUDI BRAND')

g = sns.countplot(audi_cars['model'])
audi_cars = audi_cars[(audi_cars.price < 130000)]      # removing a few apparent outliers

audi_cars = audi_cars[(audi_cars.age_months < 700)]       # small additional cleaning

sns.set_context("notebook", font_scale=1.0)

sns.pairplot(audi_cars,x_vars=['age_months'],y_vars='price',size=7)
fabia_cars = skoda_cars[skoda_cars.model =='fabia']

X = fabia_cars[['age_months','km/1000','powerPS']]

XG = audi_cars[['age_months','km/1000','powerPS']]

from sklearn.preprocessing import PolynomialFeatures     

poly = PolynomialFeatures(degree=2)

X = poly.fit_transform(X)

XG = poly.fit_transform(XG)

x_ex = poly.fit_transform([[177,150,75]])
y = fabia_cars['price']

yg = audi_cars['price']

print('Number of Skoda Fabia cars on sale is:',len(y),'and the mean price is',

      int(np.mean(y)),'Euros')

print('Number of Audi cars on sale is:',len(yg),'and the mean price is',

      int(np.mean(yg)),'Euros')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

XG_train,XG_test,yg_train, yg_test = train_test_split(XG,yg,test_size=0.25)

XG_train.shape,yg_train.shape,XG_test.shape, yg_test.shape
from sklearn.linear_model import LinearRegression  

lire= LinearRegression()
lire.fit(X_train,y_train)          

y_pred= lire.predict(X_test) 

y_pred_price = lire.predict(x_ex)

print(int(y_pred_price))

lire.fit(XG_train,yg_train)

yg_pred= lire.predict(XG_test)
from sklearn.metrics import r2_score

print('R2 score for Skoda Fabia is:',r2_score(y_test, y_pred))

print('R2 score for Audi is:',r2_score(yg_test, yg_pred))
sns.set_context("notebook", font_scale=1.0)

plt.figure(figsize=(10, 5))

plt.scatter(y_test, y_pred, s=20)

plt.title('Predicted vs. Actual for Skoda Fabia')

plt.xlabel('Actual Prices')

plt.ylabel('Predicted Prices')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])

plt.tight_layout()
sns.set_context("notebook", font_scale=1.1)

plt.figure(figsize=(10, 5))

plt.scatter(yg_test, yg_pred, s=20)

plt.title('Predicted vs. Actual for Audi')

plt.xlabel('Actual Prices')

plt.ylabel('Predicted Prices')

plt.plot([min(yg_test), max(yg_test)], [min(yg_test), max(yg_test)])

plt.tight_layout()
a4_cars = audi_cars[audi_cars.model =='a4']

XG = a4_cars[['age_months','km/1000','powerPS']]

from sklearn.preprocessing import PolynomialFeatures     

poly = PolynomialFeatures(degree=2)

XG = poly.fit_transform(XG)

x_ex = poly.fit_transform([[177,150,75]])

yg = a4_cars['price']

print('Number of Audi a4 cars on sale is:',len(yg),'and the mean price is',

      int(np.mean(yg)),'Euros')
from sklearn.model_selection import train_test_split

XG_train,XG_test,yg_train, yg_test = train_test_split(XG,yg,test_size=0.25)

XG_train.shape,yg_train.shape,XG_test.shape, yg_test.shape
from sklearn.linear_model import LinearRegression  

lire= LinearRegression()

lire.fit(XG_train,yg_train)

yg_pred= lire.predict(XG_test)

print(int(y_pred_price))
from sklearn.metrics import r2_score

print('R2 score for Audi a4 is:',r2_score(yg_test, yg_pred))
sns.set_context("notebook", font_scale=1.1)

plt.figure(figsize=(10, 5))

plt.scatter(yg_test, yg_pred, s=20)

plt.title('Predicted vs. Actual for Audi a4')

plt.xlabel('Actual Prices')

plt.ylabel('Predicted Prices')

plt.plot([min(yg_test), max(yg_test)], [min(yg_test), max(yg_test)])

plt.tight_layout()