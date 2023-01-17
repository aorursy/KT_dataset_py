import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sls # visulation tools



data = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')



data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country',

                     'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target',

                     'nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group',

                     'targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},

                      inplace=True)



data=data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed',

           'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

data['casualities']=data['Killed']+data['Wounded']

data.head(10)

data.info()
data.Country.value_counts()
