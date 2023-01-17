import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
%matplotlib inline


print(os.listdir("../input"))
df = pd.read_csv('../input/cholera-dataset/data.csv')
df.head()
df.info()
df.describe()
pp.ProfileReport(df)
# Making column names shorter
df.rename(columns = {'Number of reported cases of cholera': 'Cases', 
                          'Number of reported deaths from cholera': 'Deaths', 
                          'Cholera case fatality rate': 'Fatality rate', 
                          'WHO Region': 'Region'}, inplace = True)
df [(~df['Cases'].fillna('0').str.replace(' ','').str.isnumeric()) | (~df['Deaths'].fillna('0').str.replace('.','').str.isnumeric()) | (~df['Fatality rate'].fillna('0').str.replace('.','').str.isnumeric())]
df['Cases'] = df['Cases'].str.replace('3 5','3').str.replace(' ','').astype('float')
df['Deaths'] = df['Deaths'].str.replace('Unknown','0').str.replace('0 0','0').astype('float')
df['Fatality rate'] = df['Fatality rate'].str.replace('Unknown','0').str.replace('0.0 0.0','0').astype('float')
df[df['Fatality rate'] > 100]
df.loc[1094, 'Deaths'] = 0
df.loc[1094, 'Fatality rate'] = 0