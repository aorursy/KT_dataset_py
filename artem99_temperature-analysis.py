



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



path='/kaggle/input/temperature-readings-iot-devices/IOT-temp.csv'
data=pd.read_csv(path,sep=',')



data.head()
data.describe()
max_temp=data.temp.max()

min_temp=data.temp.min()

mean=data.temp.mean()

std=data.temp.std()



print('MAX TEMP :',max_temp)

print('MIN TEMP :',min_temp)

print('MEAN :',mean)

print('STD :',std)
data.pivot_table('temp',columns=['out/in'],aggfunc=np.mean)
data.pivot_table('temp',columns=['out/in'],aggfunc=np.max)
data.pivot_table('temp',columns=['out/in'],aggfunc=np.min)
import matplotlib.pyplot as plt

tend=data.groupby(['temp', 'out/in'])



jk=tend.size().unstack().fillna(0)

fig,az=plt.subplots(nrows=2,figsize=(12,8))

az[0].plot(jk.index,jk.Out)

az[0].legend(loc="OUT")

az[1].plot(jk.index,jk.In)

az[1].legend(loc="IN")

plt.show()
out_in=data['out/in'].value_counts()

out_in.plot(kind='bar', rot=0)


# number of packages at one moment

data.noted_date.value_counts()
# number of readings per day

data_one=data.copy()

data_one.noted_date=[x[:5]for x in data_one.noted_date.dropna()]

data_one.noted_date.value_counts()
data_one.noted_date=[x[3:5]for x in data_one.noted_date.dropna()]

f=data_one.pivot_table('temp',columns=['noted_date'],aggfunc=np.mean)

f.plot(kind='bar', rot=0)
data_day=data.copy()

data_day['MOnth']=[x[3:5]for x in data_day.noted_date.dropna()]

data_day.head()
data_day['Day']=[x[:2]for x in data_day.noted_date.dropna()]

data_day.head()

# number of unique days

data_day.Day.sort_values().unique()


# average temperature on the day of each month

f=data_day.pivot_table('temp',index=['Day'],columns=['MOnth'],aggfunc=np.mean)

f


f=f.fillna(f.mean())

f

fig,az=plt.subplots(nrows=6,figsize=(18,15))

az [ 0  ].plot(f.index,f['07'])

az[0].legend(loc="7")

 

az [ 1 ].plot(f.index,f['08'])

az[1].legend(loc="8")    



az [ 2 ].plot(f.index,f['09'])

az[2].legend(loc="9")



az [ 3 ].plot(f.index,f['10'])

az[3].legend(loc="10")



az [ 4 ].plot(f.index,f['11'])

az[4].legend(loc="11")



az [ 5 ].plot(f.index,f['12'])

az[5].legend(loc="12")

plt.show()