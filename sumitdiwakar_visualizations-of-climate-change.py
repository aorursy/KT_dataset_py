# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
##year wise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

Temp_data=pd.read_csv('../input/GlobalTemperatures.csv')

Temp_data['dt']=pd.to_datetime(Temp_data.dt)
Temp_data['year']=Temp_data['dt'].map(lambda x: x.year)
Temp_data=Temp_data.dropna()

#Calculating average year temperature
year_avg=[]
for i in range(1750,2014):
    year_avg.append(Temp_data[Temp_data['year']==i]['LandAverageTemperature'].mean())


years=range(1750,2014)

#calculating 5 years average temperatures
fiveyear=[]
for i in range(1755,2019):
    a=[]
    for j in range(i-5,i):
        a.append(Temp_data[Temp_data['year']==(j-5)]['LandAverageTemperature'].mean())
    fiveyear.append(sum(a)/float(len(a)))

#plotting graphs

plt.figure(figsize=(10,8))
plt.plot(years,np_fiveyear_avg,'r',label='Five year average temperature')
plt.plot(years,np_year_avg,'b',label='Annual average temperature')
plt.legend(loc='upper left')
plt.title('Global Average Temperature')
plt.xlabel('Years')
plt.ylabel('Temperature')
plt.show()



            