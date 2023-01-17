# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#READING DATA

df = pd.read_csv(r'../input/SEN oct-2016.csv')



### SETTING SEABORN STYLE

sns.set_style(style='darkgrid')

sns.set_palette('viridis')



### WORKING WITH DATA

df['Produccion (MWh)2'] = np.where( (df['Planta (CENCE)'] == 'Intercambio Norte') | (df['Produccion (MWh)'] < 0), np.nan, df['Produccion (MWh)'] )

temp = pd.DataFrame(  list(df['Dia'].str.split('/')), columns=['A', 'B', 'C'] )

temp['D'] = np.where( temp['B'].str.len() == 1, '0' + temp['B'], temp['B'] )

df['dia'] = temp['D']

df1 = pd.pivot_table(df, index=['dia'], values='Produccion (MWh)2', columns='Fuente', aggfunc='sum')

df1['hidro_tot'] = df1['Hidro'] + df1['Hidroelectrica']

df1['index'] = np.arange(1,32)

df1['hi'] = df1['hidro_tot']

df1['geo']  =df1['hi'] + df1['Geotermica']

df1['wi'] = df1['geo'] + df1['Eolica']

print (df1.head())
# GRAPH

plt.plot( df1['hi'], label='Hydro', )

plt.plot( df1['geo'], label='Geothermal', )

plt.plot( df1['wi'], label='Wind', )

plt.legend( bbox_to_anchor=(1.0, 0.2) )

plt.ylim( (10000, 35000) )

plt.title('Costa Rica October 2016: Main sources of Electricity Generation', fontsize=15)

plt.ylabel('MWh')

plt.xlabel('Day of October')

plt.show()