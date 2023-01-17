import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("/kaggle/input/ebola-outbreak-20142016-complete-dataset/ebola_data_db_format.csv")
df.head()
df.groupby('Indicator').sum()
df['Date']=pd.to_datetime(df['Date'])
df.sort_values(by=['Date'],inplace=True)
import pandas_profiling
df.profile_report()
df['No. of suspected cases']=df[df['Indicator']=="Cumulative number of suspected Ebola cases"]['value']

df['No. of probable cases']=df[df['Indicator']=="Cumulative number of probable Ebola cases"]['value']

df['No. of confirmed cases']=df[df['Indicator']=="Cumulative number of confirmed Ebola cases"]['value']

df['No. of confirmed, probable and suspected cases']=df[df['Indicator']=="Cumulative number of confirmed, probable and suspected Ebola cases"]['value']

df['No. of suspected deaths ']=df[df['Indicator']=="Cumulative number of suspected Ebola deaths"]['value']

df['No. of probable deaths ']=df[df['Indicator']=="Cumulative number of probable Ebola deaths"]['value']

df['No. of confirmed deaths ']=df[df['Indicator']=="Cumulative number of confirmed Ebola deaths"]['value']

df['No. of confirmed, probable and suspected deaths']=df[df['Indicator']=="Cumulative number of confirmed, probable and suspected Ebola deaths"]['value']
df.drop(['Indicator','value'],axis=1,inplace=True)
df.head()
final=df.groupby(['Date','Country']).sum().head(10)
final=final.reset_index()
final.head(10)