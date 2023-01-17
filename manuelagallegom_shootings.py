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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import seaborn as sns

data=pd.read_csv('../input/gun-violence-database/mass_shootings_2016.csv')

data['month'] = pd.DatetimeIndex(data['Incident Date']).month

plt.bar(data['month'],data['# Killed'], color= 'm')

plt.xlabel('Mes')

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])

plt.ylabel('Número de víctimas')

plt.title('Número de muertes causadas por Mass Shootings en Estados Unidos en el 2016')

plt.show()

data_=pd.read_csv('../input/gun-violence-database/mass_shootings_2015.csv')

data_['month'] = pd.DatetimeIndex(data_['Incident Date']).month

plt.bar(data_['month'],data_['# Killed'], color= 'r')

plt.xlabel('Mes')

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])

plt.ylabel('Número de víctimas')

plt.title('Número de muertes causadas por tiroteos en Estados Unidos en el 2015')

plt.show()
data1=pd.read_csv('../input/gun-violence-database/children_killed.csv')

data1['year'] = pd.DatetimeIndex(data1['Incident Date']).year  

plt.bar(data1['year'],data1['# Killed'], color='g')

plt.xlabel('Año')

plt.ylabel('Número de víctimas')

plt.title('Muertes de niños en Mass Shootings')

plt.xticks([2014,2015, 2016])

plt.show()
data2=pd.read_csv('../input/gun-violence-database/teens_killed.csv')

data2['year'] = pd.DatetimeIndex(data1['Incident Date']).year  

plt.bar(data2['year'],data2['# Killed'], color='c')

plt.xlabel('Año')

plt.ylabel('Número de víctimas')

plt.title('Muertes de adolescentes en Mass Shootings')

plt.xticks([2014,2015, 2016])

plt.show()