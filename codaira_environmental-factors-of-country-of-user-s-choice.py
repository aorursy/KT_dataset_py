# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#loading the needed data onto df

df = pd.read_csv('../input/Indicators.csv')



# choosing the required factors which determine the environmental progress of a nation

chosen_indicators=['AG.LND.FRST.ZS',\

                  'EG.FEC.RNEW.ZS',\

                  'EG.USE.COMM.FO.ZS',\

                  'EN.ATM.CO2E.KT',\

                  'ER.H2O.FWTL.K3']

df_subset = df[df['IndicatorCode'].isin(chosen_indicators)]
##country_name=input("write the name of country ")##

country_name="India"
##getting indicators of only the selected country

df_country=df_subset[df['CountryName']==country_name]



#plotting



title=country_name

ds_indicator1 = df_country[['IndicatorName','Year','Value']][df_country['IndicatorCode']==chosen_indicators[0]]

x1 = ds_indicator1['Year'].values

y1 = (ds_indicator1['Value'].values-np.array(list(ds_indicator1['Value'].values)).mean())/(np.array(list(ds_indicator1['Value'].values)).var())



ds_indicator2 = df_country[['IndicatorName','Year','Value']][df_country['IndicatorCode']==chosen_indicators[1]]

x2 = ds_indicator2['Year'].values

y2 = (ds_indicator2['Value'].values-np.array(list(ds_indicator2['Value'].values)).mean())/(np.array(list(ds_indicator2['Value'].values)).var())



ds_indicator3 = df_country[['IndicatorName','Year','Value']][df_country['IndicatorCode']==chosen_indicators[2]]

x3 = ds_indicator3['Year'].values

y3 = (ds_indicator3['Value'].values-np.array(list(ds_indicator3['Value'].values)).mean())/(np.array(list(ds_indicator3['Value'].values)).var())



ds_indicator4 = df_country[['IndicatorName','Year','Value']][df_country['IndicatorCode']==chosen_indicators[3]]

x4 = ds_indicator4['Year'].values

y4 = (ds_indicator4['Value'].values-np.array(list(ds_indicator4['Value'].values)).mean())/(np.array(list(ds_indicator4['Value'].values)).var())



ds_indicator5 = df_country[['IndicatorName','Year','Value']][df_country['IndicatorCode']==chosen_indicators[4]]

x5 = ds_indicator5['Year'].values

y5 = (ds_indicator5['Value'].values-np.array(list(ds_indicator5['Value'].values)).mean())/(np.array(list(ds_indicator5['Value'].values)).var())



plt.figure(figsize=(14,4))

    

plt.subplot(121)

plt.plot(x1,y1,label=np.unique(df_country[['IndicatorName']][df_country['IndicatorCode']==chosen_indicators[0]]))

plt.plot(x2,y2,label=np.unique(df_country[['IndicatorName']][df_country['IndicatorCode']==chosen_indicators[1]]))

plt.plot(x3,y3,label=np.unique(df_country[['IndicatorName']][df_country['IndicatorCode']==chosen_indicators[2]]))

plt.plot(x4,y4,label=np.unique(df_country[['IndicatorName']][df_country['IndicatorCode']==chosen_indicators[3]]))

plt.plot(x5,y5,label=np.unique(df_country[['IndicatorName']][df_country['IndicatorCode']==chosen_indicators[4]]))

plt.title(title)

plt.legend(bbox_to_anchor=(1.1, 1.05))


