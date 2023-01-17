## Temperature Change of India



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

plt.figure(figsize=(10,8))

plt.plot(time_period,average_temp)

plt.xlabel('years')

plt.ylabel('Average Temperature')

plt.title('Average Temperature of India over years')



#Adding Trendline



z=np.polyfit(time_period,average_temp,10)

p = np.poly1d(z)

plt.plot(time_period,p(time_period),"r--")

print("average_temp=%.6fyears+(%.6f)"%(z[0],z[1]))

plt.show()





###We can see that temperature is rising over time and since one century rise is 1 degree celcius. 







## Global Temperature Variation 



Temp_data=pd.read_csv('../input/GlobalTemperatures.csv')



Temp_data['dt']=pd.to_datetime(Temp_data.dt)

Temp_data['year']=Temp_data['dt'].map(lambda x: x.year)



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



#for plotting

np_year_avg=np.array(year_avg)

np_fiveyear_avg=np.array(fiveyear)

#plotting graphs



plt.figure(figsize=(10,8))

plt.grid()

plt.plot(years,np_fiveyear_avg,'r',label='Five year average temperature')

plt.plot(years,np_year_avg,'b',label='Annual average temperature')

plt.legend(loc='upper left')

plt.title('Global Average Temperature')

plt.xlabel('Years')

plt.ylabel('Temperature')

plt.show()



#Now clearly we can observe the rate of warming is increased in last 50 years as compared to 50 years before that.



#to see that variation its better to observe data from 1850-2013. 
## Analysing temperature difference among countries in between years



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl



Temp_data=pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')



Temp_data['dt']=pd.to_datetime(Temp_data.dt)

Temp_data['year']=Temp_data['dt'].map(lambda x: x.year)

Temp_data=Temp_data.dropna()



grouped_data=Temp_data['AverageTemperature'].groupby([Temp_data['year'],Temp_data['Country']]).mean()

grouped_dict=grouped_data.to_dict()



Temp={}

countries=Temp_data['Country'].unique()





#adding year 1913 as data for 1750 doesn't contain many countries.

firstYeardata=grouped_data[1750]

lastYeardata=grouped_data[2013]





#removing out the countries whose data is available in only one time period

for country in countries:

    if (country in firstYeardata)==True and (country in lastYeardata)==True:

        Temp[country]=(lastYeardata[country]-firstYeardata[country])



Temp1=sorted(Temp.items(),key=lambda x: x[1]) 



Countries_list=[x for x,y in Temp1]

Temp_diff=[y for x,y in Temp1]



#Plotting

#print(Countries_list)

plt.figure(figsize=(8,25))

ax=plt.subplot()

ax.barh(range(len(Countries_list)),Temp_diff)

plt.yticks(range(len(Countries_list)),Countries_list)

plt.title('Temperature Difference of countries')

plt.xlabel('Temp. Diff')

plt.show()
