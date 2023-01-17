# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib

from subprocess import check_output

from pylab import fill_between

#print(check_output(["ls", "../input"]).decode("utf8"))

country = pd.read_csv('../input/Country.csv')

country_notes = pd.read_csv('../input/CountryNotes.csv')

indicators = pd.read_csv('../input/Indicators.csv')

df_elec_rural = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.ACCS.RU.ZS')]

df_elec_urban = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.ACCS.UR.ZS')]

df_elec_pop = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

fig = plt.figure()



plt.plot(df_elec_rural.Year,df_elec_rural.Value,'o-',label='Rural')

plt.plot(df_elec_urban.Year,df_elec_urban.Value,'o-',label='Urban')

plt.plot(df_elec_pop.Year,df_elec_pop.Value,'o-',label='General')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Population',  fontsize=14)

plt.title('Access to Electricity', fontsize=14)

plt.grid(True)

#print(indicators)

# Any results you write to the current directory are saved as output.
df_elec_fosl = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_elec_hydro = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.HYRO.ZS')]

df_elec_nucl = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.NUCL.ZS')]

df_elec_rnwx = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)



fig = plt.figure()



plt.plot(df_elec_fosl.Year,df_elec_fosl.Value,label='Fossil Fuels')

plt.plot(df_elec_hydro.Year,df_elec_hydro.Value,label='Hydroelectric')

plt.plot(df_elec_nucl.Year,df_elec_nucl.Value,label='Nuclear')

plt.plot(df_elec_rnwx.Year,df_elec_rnwx.Value,label='Renewable')





#fill_between(df_elec_fosl.Year,df_elec_fosl.Value,0,alpha=0.5)

#fill_between(df_elec_hydro.Year,df_elec_hydro.Value,0,alpha=0.5)

#fill_between(df_elec_nucl.Year,df_elec_nucl.Value,0,alpha=0.5)

#fill_between(df_elec_rnwx.Year,df_elec_rnwx.Value,0,alpha=0.5)

#fill_between(df_elec_rnwx.Year,df_elec_rnwx.Value,0,alpha=0.5)

#fill_between(x,y2,0,color='magenta')

#fill_between(x,y3,0,color='red')



plt.legend(loc=2, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Total Energy Produce',  fontsize=14)

plt.title('Energy Mix in the India (1971-2012)', fontsize=18)