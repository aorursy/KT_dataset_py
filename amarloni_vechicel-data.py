import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
data = pd.read_csv("/kaggle/input/epa-vehicle-dataset-19802021/vehicles.csv")

data.head()
data.describe()
data.columns.unique()
data.info()
data.isna().any().sum()
data.info()
data = data.drop(['guzzler','trans_dscr', 'tCharger', 'sCharger', 'atvType', 'fuelType2', 'rangeA',

       'evMotor', 'mfrCode', 'c240Dscr', 'charge240b', 'c240bDscr', 'startStop'], axis = 1)
data = data.dropna(axis =0)
data.info()
data.head()
data['trany']= data['trany'].map({'Manual 5-spd':'Man_5S', 'Automatic 3-spd': 'Auto_3S', 'Automatic 4-spd':'Auto_4S',

       'Automatic 5-spd':'Auto_5S', 'Manual 4-spd':'Man_4S', 'Manual 3-spd':'Man_3S', 'Manual 6-spd':'Man_6S',

       'Automatic (S5)':'Auto_5S', 'Automatic (variable gear ratios)':'Automatic',

       'Automatic 6-spd':'Auto_6S', 'Automatic (S6)':'Auto_6S','Automatic (S4)':'Auto_4S',

       'Automatic 7-spd':'Auto_7S', 'Automatic (S7)':'Auto_7S', 'Automatic (S8)':'Auto_8S',

       'Automatic (AM5)':'Auto_5S', 'Automatic (AM6)':'Auto_6S', 'Automatic (AV-S7)':'Auto_7S',

       'Automatic (AV-S6)':'Auto_6S', 'Automatic (AM7)':'Auto_7S', 'Manual 4-spd Doubled':'Man_4S',

       'Manual 7-spd':'Man_7S', 'Automatic (L4)':'Auto_L4', 'Automatic (L3)':'Auto_L3',

       'Automatic (AV-S8)':'Auto_8S', 'Automatic 8-spd':'Auto_8S', 'Automatic (A1)':'Auto_A1',

       'Automatic (AM-S6)':'Auto_6S', 'Automatic (AM-S7)':'Auto_7S', 'Automatic 9-spd':'Auto_9S',

       'Automatic (S9)':'Auto_9S', 'Automatic (AM-S8)':'Auto_8S', 'Automatic (AM8)':'Auto_8S',

       'Automatic (AM-S9)':'Auto_9S', 'Automatic (S10)':'Auto_10S', 'Automatic (AV-S10)':'Auto_10S',

       'Automatic 10-spd':'Auto_10S', 'Automatic (A2)':'Auto_A2'})
data.trany.unique()
data.drive.unique()
data['drive']=  data['drive'].map({'4-Wheel or All-Wheel Drive':'4-Wheel Drive', 'All-Wheel Drive':'4-Wheel Drive',

       'Part-time 4-Wheel Drive':'4-Wheel Drive'})
data.fuelType.unique()
data['fuelType']=data['fuelType'].map({'Regular':'Gasoline', 'Gasoline or natural gas':"Gasoline",

       'Gasoline or E85':'Gasoline', 'Premium or E85':'Premium'})
data.make.unique()
data.VClass.unique()
data['VClass']=data['VClass'].map({'Special Purpose Vehicles/4wd':'Special Purpose Vehicle 4WD', 

                                   'Special Purpose Vehicles':'Special Purpose Vehicle', 'Special Purpose Vehicle 2WD':'Special Purpose Vehicle/2wd'})


plt.figure(figsize=(10,5))

sns.lineplot(x='year', y='cityA08', data=data)

plt.title('City MPG trend for fule type 2')

plt.ylabel('MPG')

plt.show
plt.figure(figsize=(10,5))

sns.lineplot(x='year', y='barrels08', data=data)

plt.title('Annual Petroleum consumtion in barrels for type1 fuel')

plt.ylabel('Annual Barrels consumption')

plt.show
plt.figure(figsize=(10,5))

sns.lineplot(x='year', y='barrelsA08', data=data)

plt.title('Annual Petroleum consumtion in barrels for type2 fuel')

plt.ylabel('Annual Barrels consumption')

plt.show
data.groupby(['make'])['barrels08','year'].size().sort_values(ascending = False)

data.groupby(['make','VClass','cylinders']).size()