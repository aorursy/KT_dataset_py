# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
gdp=pd.read_excel('/kaggle/input/-gdp-growth-per-year/gdp-growth.xls')

gdp
gdp[2::]
gdp = gdp[(gdp['Data Source'] == 'Turkey')  | (gdp['Data Source'] == 'Germany') |  (gdp['Data Source'] == 'United States')]
gdp.drop(columns =['World Development Indicators', 'Unnamed: 2',

       'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',

       'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12',

       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16',

       'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20',

       'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24',

       'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28',

       'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32',

       'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36',

       'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40',

       'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44',

       'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48',

       'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52',

       'Unnamed: 53', 'Unnamed: 63'], inplace = True)
gdp
gdp.rename(columns = {'Data Source':'Year', 'Unnamed: 54' : '2010', 'Unnamed: 55': '2011', 'Unnamed: 56': '2012', 'Unnamed: 57': '2013', 'Unnamed: 58': '2014', 'Unnamed: 59': '2015', 'Unnamed: 60': '2016', 'Unnamed: 61': '2017', 'Unnamed: 62': '2018'}, inplace= True)
gdp.set_index(keys = ['Year'], inplace = True)

gdp
gdp = gdp.T

gdp
gdp.plot(colormap= 'Set3', figsize = (10,6), lw = 4);

plt.xlabel('Years')

plt.ylabel('GDP Growth %')

plt.title('GDP Growth of Some Countries')
unemp = pd.read_excel('/kaggle/input/unemployment-rate/data1.xls')

unemp
unemp = unemp[(unemp['Data Source'] == 'Turkey')  | (unemp['Data Source'] == 'Germany') |  (unemp['Data Source'] == 'United States')]

unemp
unemp.drop(columns = ['World Development Indicators', 'Unnamed: 2',

       'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',

       'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12',

       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16',

       'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20',

       'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24',

       'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28',

       'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32',

       'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36',

       'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40',

       'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44',

       'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48',

       'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52',

       'Unnamed: 53', 'Unnamed: 63'], inplace = True)

unemp
unemp.rename(columns = {'Data Source':'Year', 'Unnamed: 54' : '2010', 'Unnamed: 55': '2011', 'Unnamed: 56': '2012', 'Unnamed: 57': '2013', 'Unnamed: 58': '2014', 'Unnamed: 59': '2015', 'Unnamed: 60': '2016', 'Unnamed: 61': '2017', 'Unnamed: 62': '2018'}, inplace= True)
unemp.set_index(keys = ['Year'], inplace = True)
unemp = unemp.T

unemp
unemp.plot(colormap= 'Set3', figsize = (10,6), lw = 4);

plt.xlabel('Years')

plt.ylabel('Unemployment Rate')

plt.title('Unemploymant Rate % Per Year')
fig, ax1= plt.subplots(figsize=(15,5))

ax1.plot(gdp.index, gdp['Germany'],'-s' ,label ='Germany GDP') # -s means square.

ax1.plot(gdp.index, gdp['Turkey'], '-s', label = 'Turkey GDP')

ax1.plot(gdp.index, gdp['United States'],'-s', label = 'US GDP')

ax1.set_xlabel('Year')

ax1.set_ylabel('GDP Growth %')

ax1.tick_params('y')

ax1.legend(loc='lower left')



 

ax2 = ax1.twinx() # observing datasets in same years so we are using same x labels

ax2.plot(unemp.index, unemp['Germany'], '-x', label ='Germany UNEMP')

ax2.plot(unemp.index, unemp['Turkey'], '-x', label = 'Turkey UNEMP')

ax2.plot(unemp.index, unemp['United States'],'-x', label = 'US UNEMP')

ax2.set_ylabel('Unemployment Rate Per Year')

ax2.tick_params('y')

ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85)) # You can adjust your legend location with bbox_to_anchor.

plt.title('% GDP Growth vs Unemployment Rate %')

fig.tight_layout()

plt.show()