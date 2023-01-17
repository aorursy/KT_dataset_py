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
import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt 

import os
import io

reservoir_levels = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv')

reservoir_rainfall = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv')

reservoir_levels.info() 

reservoir_rainfall.info()
reservoir_levels.head()
reservoir_levels.describe()

reservoir_levels.isnull().sum()
reservoir_rainfall.describe()

reservoir_rainfall.isnull().sum()
level=reservoir_levels.copy()

rain=reservoir_rainfall.copy()

level['POONDI'].hist()
level['POONDI'].plot(figsize=(15,7))

plt.ylabel('Rain Level in Poondi')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

ax1.plot(rain['POONDI'])

ax2.plot(rain['CHOLAVARAM'])
rain['Total']=rain['POONDI']+rain['CHOLAVARAM']+rain['REDHILLS']+rain['CHEMBARAMBAKKAM']

rain['Date'] = pd.to_datetime(rain['Date'])

#rain.set_index('Date', inplace=True)

#rain.reset_index()'

#rain.head()
#rain.reset_index()

rain['Year']=rain['Date'].dt.year

rain['Month']=rain['Date'].dt.month

rain.head()
grp_by_year = rain.groupby('Year')['Total'].sum()

grp_by_year.plot(kind='bar');

plt.ylabel('Level of Water over respective years!');

plt.xticks(rotation=60);

#grp_by_year.head()
rain_POONDI = rain.loc[:,['POONDI','Year','Total']]

rain_REDHILLS = rain.loc[:,['REDHILLS','Year','Total']]

rain_CHEMBARAMBAKKAM = rain.loc[:,['CHEMBARAMBAKKAM','Year','Total']]

rain_CHOLAVARAM = rain.loc[:,['CHOLAVARAM','Year','Total']]
plt.figure(figsize=(18,12))

plt.subplot(2,2,1)

grp_poondi = rain_POONDI.groupby('Year')['Total'].sum()

grp_poondi.plot(kind='barh')

plt.title("POONDI's Reservoir's performance!");



plt.subplot(2,2,2)

grp_redhills = rain_REDHILLS.groupby('Year')['Total'].sum()

grp_redhills.plot(kind='barh');

plt.title("REDHILLS's Reservoir's performance!");



plt.subplot(2,2,3)

grp_chem = rain_CHEMBARAMBAKKAM.groupby('Year')['Total'].sum()

grp_chem.plot(kind='barh');

plt.title("CHEMBARAMBAKKAM's Reservoir's performance!");



plt.subplot(2,2,4)

grp_chol = rain_CHOLAVARAM.groupby('Year')['Total'].sum()

grp_chol.plot(kind='barh');

plt.title("CHOLAVARAM's Reservoir's performance!");
plt.figure(figsize=(18,12))

plt.subplot(1,2,1)

TotalvsPoondi = rain_POONDI.groupby('Year')[['POONDI','Total']].sum()

TotalvsPoondi.plot(kind='bar',ax=plt.gca())



plt.subplot(1,2,2)

TotalvsPoondi.plot(kind='bar',stacked=True ,ax=plt.gca())