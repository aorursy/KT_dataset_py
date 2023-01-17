%matplotlib inline

import os 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt 

import numpy as np

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')
train.info()
gotham_city = train.plot(kind='scatter', figsize=(27,20), grid = False, x='dropoff_longitude', y='dropoff_latitude',color='white',xlim=(-74.110336,-73.83),ylim=(40.659212, 40.878145),s=.02,alpha=.6)

gotham_city.set_axis_bgcolor('black')
gotham_city = train.plot(kind='scatter', figsize=(27,20), grid = False, x='pickup_longitude', y='pickup_latitude',color='white',xlim=(-74.110336,-73.83),ylim=(40.659212, 40.878145),s=.02,alpha=.6)

gotham_city.set_axis_bgcolor('black')