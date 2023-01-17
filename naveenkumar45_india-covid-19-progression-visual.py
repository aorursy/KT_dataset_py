# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd 
from geopandas import gpd 
import matplotlib.pyplot as plt
import time 
df = pd.read_csv('covid_19_india.csv')
#print("df",df['State'].unique())
dat=list(df['Date'].unique())
sta=list(df['State'].unique())
ds=[]
print(df.shape)
print(len(dat))
print(len(sta))
for h in range(len(dat)):
    #print("df1",df['Date'].unique())
    df1=df.loc[df['Date']==dat[h]]
    for h1 in range(len(sta)):
        df2=df1.loc[df['State']==sta[h1]]
        if df2.shape[0]==0:
            d={}
            d['Date']=dat[h]
            d['Time']=0
            d['State']=sta[h1]
            d['ConfirmedIndianNational']=0
            d['ConfirmedForeignNational']=0
            d['Cured']=0
            d['Deaths']=0
            d['Confirmed']=0
            ds.append(d)
        df2=None
    df1=None
df3=pd.DataFrame(ds)
df4= pd.concat([df, df3],ignore_index=True)
print(df4.shape)
for k in range(len(dat)):
    df5=df4.loc[df4['Date']==dat[k]]
    print(df5['Date'].unique())
    fp = "Igismap/Indian_States.shp"
    map_df = gpd.read_file(fp)
    merged = map_df.set_index('st_nm').join(df5.set_index('State'))
    merged.fillna(0)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axis('off')
    gy="Niyata Infotech_Analysis "+str(dat[k])+"Confirmed"
    ax.set_title(gy, fontdict={'fontsize': '14', 'fontweight' : '2'})
    merged.plot(column='Confirmed', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    fig.savefig("First_Level_Report/"+str(k)+".png", dpi=100)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
