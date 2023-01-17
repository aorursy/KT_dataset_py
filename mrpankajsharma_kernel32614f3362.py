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
data=pd.read_csv("/kaggle/input/wildlife-strikes/database.csv")
df=data.loc[:,['Species Name', 'Species Quantity', 'Flight Impact', 'Fatalities','Injuries', 

       'Aircraft Damage', 'Radome Strike', 'Radome Damage',

       'Windshield Strike', 'Windshield Damage', 'Nose Strike', 'Nose Damage',

       'Engine1 Strike', 'Engine1 Damage', 'Engine2 Strike', 'Engine2 Damage',

       'Engine3 Strike', 'Engine3 Damage', 'Engine4 Strike', 'Engine4 Damage',

       'Engine Ingested', 'Propeller Strike', 'Propeller Damage',

       'Wing or Rotor Strike', 'Wing or Rotor Damage', 'Fuselage Strike',

       'Fuselage Damage', 'Landing Gear Strike', 'Landing Gear Damage',

       'Tail Strike', 'Tail Damage', 'Lights Strike', 'Lights Damage',

       'Other Strike', 'Other Damage']]
def func(value):

    if value=='1':

        return 1

    elif value=='2-10':

        return 5

    elif value=='11-100':

        return 50

    elif value=='Over 100':

        return 100 

df['Species Quantity']=df['Species Quantity'].apply(func)        

        
df['Flight Impact']=df['Flight Impact'].apply(lambda value:0 if value=='NONE' else 1).fillna(0)

df['Fatalities']=df['Fatalities'].fillna(0)

df['Injuries']=df['Injuries'].fillna(0)



df['Species Name']=df['Species Name'].fillna('UNKNOWN')

df['Species Name']=df[~df['Species Name'].str.contains('UNKNOWN')]
df=df.dropna()
df['Cumulative Damage']=df.sum(axis = 1)
df.drop(['Species Quantity', 'Flight Impact', 'Fatalities','Injuries', 

       'Aircraft Damage', 'Radome Strike', 'Radome Damage',

       'Windshield Strike', 'Windshield Damage', 'Nose Strike', 'Nose Damage',

       'Engine1 Strike', 'Engine1 Damage', 'Engine2 Strike', 'Engine2 Damage',

       'Engine3 Strike', 'Engine3 Damage', 'Engine4 Strike', 'Engine4 Damage',

       'Engine Ingested', 'Propeller Strike', 'Propeller Damage',

       'Wing or Rotor Strike', 'Wing or Rotor Damage', 'Fuselage Strike',

       'Fuselage Damage', 'Landing Gear Strike', 'Landing Gear Damage',

       'Tail Strike', 'Tail Damage', 'Lights Strike', 'Lights Damage',

       'Other Strike', 'Other Damage'],axis=1,inplace=True)
df=df.groupby(['Species Name']).sum()
df.sort_values(['Cumulative Damage','Species Name'],axis=0,ascending=False,inplace=True)

df