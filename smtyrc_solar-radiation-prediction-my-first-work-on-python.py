# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dt=pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')



dt.info()

dt.columns

dt.head(10)
dt.corr ()  
f,ax = plt.subplots(figsize=(14, 14)) 

sns.heatmap(dt.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)   
dt.Radiation.plot(kind = 'line', color = 'black',label = 'Radiation',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')

dt.Temperature.plot(kind = 'line', color = 'red',label = 'Temperature',linewidth=2,alpha = 0.5,grid = True,linestyle = ':')

plt.rcParams["figure.figsize"] = (13,11)

plt.legend(loc='upper right') 

plt.xlabel('x axis')                           

plt.ylabel('y axis')

plt.title('Line Plot')                         

plt.show()
dt.plot(kind='scatter', x='Pressure', y='Humidity',color = 'red',alpha = 0.5, )   

plt.scatter(dt.Pressure, dt.Humidity , color = 'blue', alpha = 0.5)     

plt.xlabel('Pressure')              

plt.ylabel('Humidity')

plt.title('Pressure - Humidity Scatter Plot')

plt.show()
dt.Temperature.plot(kind = 'hist',bins = 80,figsize = (12,12))    

plt.show()

dt.columns

df2 = pd.DataFrame(dt, columns = ['Data' , 'Radiation' , 'Temperature', 'Pressure' , 'Humidity']) 

df2
dt_dict = dt.to_dict()

dt_dict['Temperature']

'Humidity' in dt_dict
del dt_dict['UNIXTime']

dt_dict['UNIXTime']
series = dt['Radiation'] 

series
dt[np.logical_and(dt['Temperature']>48, dt['Humidity']>80 )]



dt_dict 



dt_list=dt_dict.items()

dt_list



for index, value in enumerate(dt_list):

          print(index," : ",value)

print('')    





for index,value in dt[['Speed']][20:30].iterrows():    

    print(index," : ",value)