#import packages

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import pandas_datareader as pdr

from pandas_datareader import wb
#get data from World Bank Indicators

df = wb.download(indicator=['NY.GDP.MKTP.KD.ZG','NY.GDP.PCAP.CD','NY.GDP.MKTP.CD','SP.RUR.TOTL.ZS'], country=['KEN', 'MOZ', 'ZMB', 'ZWE', 'AFG', 'AGO', 'BDI'], start=1990, end=2000)
#print first 5 rows

df.head()
# reset index

df =df.reset_index()

df.head()
#rename columns

df.columns = ['country', 'year','gdp_growth', 'gdp_cap', 'gdp','rural_pop']
#get info about the dataset

df.info()
# export the df to csv



##kaggle path to export the df

df.to_csv('/kaggle/working/wb.csv',index=False)