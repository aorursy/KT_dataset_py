# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

Temp_data=pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
#reading file using pandas
print(Temp_data)
India_temp=Temp_data[Temp_data['Country']=='India'].dropna()
#sorting out data of India

India_temp['dt']=pd.to_datetime(India_temp.dt)
India_temp['year']=India_temp['dt'].map(lambda x: x.year)

#This is done for 100 years of data to see change in temperature in last one century. 
print(India_temp['year'].max())   #2013
time_period=range(1913,2013)

average_temp=[]
for year in time_period:
    average_temp.append(India_temp[India_temp['year']==year]['AverageTemperature'].mean())

#plotting
plt.figure(figsize=(8,6))
plt.plot(time_period,average_temp)
plt.xlabel('years')
plt.ylabel('Average Temperature')
plt.title('Average Temperature of India over years')

#Adding Trendline

#z=np.polyfit(time_period,average_temp,10)
#p = np.poly1d(z)
#plt.plot(time_period,p(time_period),"r--")
#print("average_temp=%.6fyears+(%.6f)"%(z[0],z[1]))
#plt.show()


###We can see that temperature is rising over time and since one century rise is 1 degree celcius. 

print("Hello Notepad")