import numpy as np

import sklearn

import pandas as pd

import matplotlib as plt

import seaborn as sns

import matplotlib.pyplot as plt

import string

import re

import random



import datetime as dt

from datetime import datetime

%matplotlib inline  

metadata = pd.read_csv('../input/database.csv')
metadata.head(2).T
metadata.isnull().sum()
Data = pd.read_csv("../input/database.csv",usecols=[0,1,2,3,4,5,7,10,11,12,18,19,20,21,22,23])
Name_of_satellite = Data['Official Name of Satellite']

Name_of_satellite.value_counts().head(10)
Operator_Owner = Data['Operator/Owner'] 

Operator_Owner.value_counts().head(20).plot(kind='bar')

plt.title('Satellite Using By Differnt Users')

Country_Operator_Owner = Data['Country of Operator/Owner']

Country_Operator_Owner.value_counts().head(20).plot(kind='bar')

plt.title('Satellite Overview Based On Each Country')

Users = Data['Users']

Users.value_counts().head(4).plot(kind='bar')

plt.title('Uses of Satellites In World')

Purpose = Data['Purpose']

Purpose.value_counts().head(10)
Class_of_orbit = Data['Class of Orbit']

Class_of_orbit.value_counts().head(4).plot(kind='bar')

plt.title('Satellites vs Class of orbit')

Data['date'] = pd.to_datetime(Data['Date of Launch'])

Data['year'], Data['month'] = Data['date'].dt.year, Data['date'].dt.month

date_count = Data.groupby('date').date.count()

values = date_count.values

dates = date_count.index



mean = sum(values)/len(date_count)

variance = np.sqrt(sum((values-mean)**2)/len(date_count))



plt.bar(dates, values)

plt.title('Number of Lanuches Per year')

plt.xticks(rotation='70')

plt.show()

Data.groupby(['Country of Operator/Owner','Purpose'])['Users'].size()
Data.groupby(['Launch Site']).size()