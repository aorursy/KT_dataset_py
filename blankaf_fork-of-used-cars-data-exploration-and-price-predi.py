import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
prep_cars=pd.read_csv('../input/autos.csv', encoding='latin_1')

columns_to_keep = ['brand','model','vehicleType','yearOfRegistration',

                   'monthOfRegistration','kilometer','powerPS','fuelType',

                   'gearbox','abtest','notRepairedDamage','price']

prep_cars = prep_cars[columns_to_keep]

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

prep_cars.head(3)
print('Number of listed cars is: ', len(prep_cars)) 
def age_count(yr, mt):                    # counting the age of the car as the time from

    return (2015-yr)*12+(12-mt)+3        # registration to April,1,2016

                                        # introducing a new attribute = car age in months  

prep_cars['age_months'] = age_count(prep_cars['reg_year'],prep_cars['reg_month'])        
prep_cars['km/1000'] = prep_cars['kilometer']/1000       # scaled feature in a new column

prep_cars.head(2)
columns_to_keep = ['brand','model','car_type','reg_year','age_months','km/1000',

                   'powerPS','fuelType','gearbox','abtest','damage','price']

prep_cars = prep_cars[columns_to_keep]

sel_cars = prep_cars[(prep_cars.age_months >= 0)&(prep_cars.price <= 150000)&

                     (prep_cars.price > 100)]

sel_cars.head(3)                   # removing records with invalid or very extreme values
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
sns.set_context("notebook",font_scale=1.2)

plt.figure(figsize=(13,5))

plt.title('MODELS DISTRIBUTION FOR AUDI BRAND')

g = sns.countplot(audi_cars['model'])
skoda_cars = skoda_cars[(skoda_cars.age_months < 400)]   # removing a few apparent outliers

audi_cars = audi_cars[(audi_cars.age_months < 600)] 

skoda_models = skoda_cars.model.unique()                  # list of models by SKODA

audi_models = audi_cars.model.unique()                    # list of models by AUDI
audi_cars.drop('brand', axis=1)

audi_cars=audi_cars.groupby('model').agg({'price':np.mean}).sort_values(by='price').astype(int)

ta = audi_cars.T                             # average price per model sorted from lowest

ta                                                # transposed table for better view
sns.set_context("notebook",font_scale=1.2)

plt.figure(figsize=(13,5))

plt.title('Model mean price for AUDI')

g = sns.barplot( x=audi_models, y='price', data=audi_cars )
pt_skoda=skoda_cars.pivot_table(values='price',index='reg_year',columns='model',

                                aggfunc=(np.mean)).round()

skoda_cars.head()                               

pt_skoda.fillna(0, inplace=True)   

pt_skoda                    # pivot table for mean price by model for year of registration
import matplotlib.pyplot as plt

import matplotlib.cm as cm 

sns.set_context("notebook",font_scale=1.1)

pt_skoda=pt_skoda[pt_skoda.index>2006].T

yticks = pt_skoda.index

hm = sns.heatmap(pt_skoda,square=True)

plt.yticks(rotation=0) 

plt.title('Mean prices in Euros for SKODA per model for year of registration')

plt.show()