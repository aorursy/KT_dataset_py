import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from __future__ import division
%config InlineBackend.figure_format = 'svg'

print(os.listdir("../input"))
xl = pd.ExcelFile('../input/AirQuality.xlsx')
xl.sheet_names
df = xl.parse('Sheet1')
df.head()
df_pm25 = df[df['Pollutants']=='PM2.5']
df_no2 = df[df['Pollutants']=='NO2']
df_so2 = df[df['Pollutants']=='SO2']
df_co = df[df['Pollutants']=='CO']
df_ozone = df[df['Pollutants']=='OZONE']
df.shape
plt.figure(figsize=(10,6))
print (df['State'].value_counts())
ax = sns.countplot(x='State', data=df);
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
fig, axes = plt.subplots(nrows=5,ncols=4,figsize=(17,20))
space = 0.25
lspace = 0.24
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1.25);
i = 0
for state in df['State'].unique():
    ax1 = sns.countplot(data=df[df['State']==state],x='city',ax=axes.flatten()[i]);
    ax1.set_title(state)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right");
    i = i + 1
df_pm25.head()
all_df_list = [df_pm25,df_no2,df_so2,df_co,df_ozone]
all_df_poll_list = ['PM2.5','NO2','SO2','CO','OZONE']

i=0
for df_poll,df_name in zip(all_df_list,all_df_poll_list):
    print (df_name)
    print (df_poll[['Avg','Max','Min']].describe())
    print('\n')