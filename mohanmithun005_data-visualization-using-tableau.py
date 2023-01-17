import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%cd /kaggle/input/
df = pd.read_csv('chennai_reservoir_levels.csv')

df.sample(5)
dfrain = pd.read_csv('chennai_reservoir_rainfall.csv')

dfrain.sample(5)

col=dfrain.columns

result = pd.merge(df,

                 dfrain[col],

                 on='Date')

result.head()
result['sum_mfct']=result['POONDI_x']+result['CHOLAVARAM_x']+result['REDHILLS_x']+result['CHEMBARAMBAKKAM_x']

result['sum_rainfall']=result['POONDI_y']+result['CHOLAVARAM_y']+result['REDHILLS_y']+result['CHEMBARAMBAKKAM_y']

result.head()
result['year'] = pd.DatetimeIndex(result['Date']).year

result['month'] = pd.DatetimeIndex(result['Date']).month

result.head()
summer_data=result[(result['month'] == 5) | (result['month'] == 6)| (result.month == 7)|(result.month==8)]



monsoon_data=result[(result['month'] == 10) | (result['month'] == 11)| (result.month == 12)|(result.month==9)]



summer_data