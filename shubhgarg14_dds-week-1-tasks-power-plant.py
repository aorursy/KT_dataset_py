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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
p1_gd=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

p1_ws=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

p2_gd=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

p2_ws=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

print('\t\t\t\t\t***Loading data from CSV files done***')



p1_gd.columns
p1_ws.columns
p2_gd.columns
p2_ws.columns
p1_gd.nunique()
p1_ws.nunique()
p2_gd.nunique()
p2_ws.nunique()
p1_gd.describe()
p1_ws.describe()
p2_gd.describe()
p2_ws.describe()
#for plant 1

p1_gd['DAILY_YIELD'].mean()
#for plant 2

p2_gd['DAILY_YIELD'].mean()
p1_ws['DATE_TIME']=pd.to_datetime(p1_ws['DATE_TIME'],format='%d-%m-%Y %H:%M:%S')

p1_ws.info()
p2_ws['DATE_TIME']=pd.to_datetime(p2_ws['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

p2_ws['DATE']=pd.to_datetime(p2_ws['DATE_TIME'].dt.date)

p2_ws.info()
d=p1_ws.groupby(['DATE']).sum()

d['IRRADIATION']

e=p2_ws.groupby(['DATE']).sum()

e['IRRADIATION']

p1_ws['AMBIENT_TEMPERATURE'].max()
p2_ws['AMBIENT_TEMPERATURE'].max()
p1_ws['MODULE_TEMPERATURE'].max()
p2_ws['MODULE_TEMPERATURE'].max()
#plant 1

p1_gd['SOURCE_KEY'].nunique()
#plant 2

p2_gd['SOURCE_KEY'].nunique()
p1_gd
p1_gd.groupby(['DATE']).max().DC_POWER
p1_gd.groupby(['DATE']).max().AC_POWER
p2_gd.groupby(['DATE']).max().DC_POWER
p2_gd.groupby(['DATE']).max().AC_POWER
p1_gd[p1_gd['DC_POWER']==p1_gd['DC_POWER'].max()]['SOURCE_KEY']
p1_gd[p1_gd['AC_POWER']==p1_gd['AC_POWER'].max()]['SOURCE_KEY']
p2_gd[p2_gd['DC_POWER']==p2_gd['DC_POWER'].max()]['SOURCE_KEY']
p2_gd[p2_gd['AC_POWER']==p2_gd['AC_POWER'].max()]['SOURCE_KEY']
a=p1_gd.groupby(['SOURCE_KEY']).sum().DC_POWER

a.sort_values()
b=p1_gd.groupby(['SOURCE_KEY']).sum().AC_POWER

b.sort_values()
c=p2_gd.groupby(['SOURCE_KEY']).sum().DC_POWER

c.sort_values()
d=p2_gd.groupby(['SOURCE_KEY']).sum().AC_POWER

d.sort_values()
#Total Number of rows should be:

# 22 inverters * 34 days * 24 hours * 4 set of data per hour

#Total should be:

print('Total: ',22*34*24*4)
#But we have

p1_gd.count()

print('That is: ',71808-68778,'less')
#for plant 2

p2_gd.count()
print('That is: ',71808-67698,'less')