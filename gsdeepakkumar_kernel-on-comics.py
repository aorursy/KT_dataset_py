## importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

os.listdir()
## import dataset:

kaggle=1

if kaggle==1:

    dc=pd.read_csv('../input/dc-wikia-data.csv')

    marvel=pd.read_csv('../input/marvel-wikia-data.csv')

else:

    dc=pd.read_csv('dc-wikia-data.csv')

    marvel=pd.read_csv('marvel-wikia-data.csv')

    
## Glimpse at the data:

dc.head()
marvel.head()
marvel['WORLD']='Marvel'

dc['WORLD']='DC'
marvel.info()
dc.info()
print(f'There are {dc.page_id.nunique()} DC Characters and {marvel.page_id.nunique()} marvel characters')
data=pd.concat([marvel,dc])
data.shape
data.isnull().sum().sort_values(ascending=False)
sex=data['SEX'].value_counts()

print('% Distribution of the characters based on gender')

print(sex/len(data['SEX'])*100)