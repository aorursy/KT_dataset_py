"""ACE (accumulated cyclone energy) Data source from

    Global Tropical Cyclone Activity 

    Dr. Ryan N. Maue

    https://policlimate.com/tropical/

    

    sea surface temperature from pre-industrial times



    source :

    US EPA climate change indicators

    https://www.epa.gov/climate-indicators/climate-change-indicators-sea-surface-temperature    

"""



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

ACE_FILE = '../input/tropical-cyclone-ace/tropical_storm_ACE.txt'

SS_FILE = '../input/tropical-cyclone-ace/sea_surface_temperatures.txt'
ace = pd.read_csv(ACE_FILE,delimiter='\t',parse_dates=['MONTH'])

ss = pd.read_csv(SS_FILE)
ace.head()
ss.head()
period = 4 ; base_year = [1970,1878]

ace['year_group'] = ace.apply(lambda x: 

        base_year[0]+period*(math.floor((x['MONTH'].year-base_year[0])/period)),axis=1)

ss['year_group'] = ss.apply(lambda x: 

        base_year[1]+period*(math.floor((x['Year']-base_year[1])/period)),axis=1)
ace_pvt = pd.pivot_table(ace,index=['year_group'],values=['GLOBAL'],aggfunc='mean').reset_index()

ss_pvt = pd.pivot_table(ss,index=['year_group'],values=['Annual anomaly'],aggfunc='mean').reset_index()

df = pd.merge(ace_pvt,ss_pvt,on='year_group')

df.rename(columns={'GLOBAL':'ACE','Annual anomaly':'sea surface temp+'},inplace=True)
df
import seaborn as sb

sb.scatterplot(x='sea surface temp+',y='ACE',data=df)
import matplotlib.pyplot as plt

#sb.set(rc={'figure.figsize':(11, 4)})

f, axes = plt.subplots(2,1,figsize=(16,8))

sb.lineplot(x='year_group',y='ACE',label='ACE',data=df,ax=axes.flat[0])

sb.lineplot(x='year_group',y='sea surface temp+',label='sea temp',data=df,ax=axes.flat[1])