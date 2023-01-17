import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,5)
train = pd.read_csv('../input/train.csv', index_col='row_id')
train.describe()
test = pd.read_csv('../input/test.csv', index_col='row_id')
test['place_id'] = -1
#test.head()
test.describe()
df = pd.concat([train, test])
idx_test = (df.place_id == -1)
print(df.head())
print(df.tail())
df.describe()
checkins, bins = np.histogram(df.time, bins=range(0,df.time.max()+60,60))
fft = np.fft.fft(checkins)
plt.xlim(10,3000)
plt.ylim(10**3,10**8)
for x in [100, 200, 300, 400]:
    plt.axvline(x,color='red', ls='--')
for x in [699,699*2, 699*3]:
    plt.axvline(x,color='green', ls='--')
for x in [499, 599, 799, 899]:
    plt.axvline(x,color='orange', ls='--')
for x in [1298]:
    plt.axvline(x,color='black', ls='--')    
plt.loglog(np.sqrt(fft * fft.conj()).real)
plt.xlabel('Number of events')
rng = pd.date_range('1/3/2013',periods=(100*7-1)*24,freq='H')
checkin_sim = pd.DataFrame(index=rng)
checkin_sim['open'] = 0
checkin_sim['dayofweek'] = rng.dayofweek
checkin_sim['month'] = rng.month
checkin_sim['day'] = rng.day
checkin_sim['hour'] = rng.hour
checkin_sim.ix[(checkin_sim.hour>8) & (checkin_sim.hour<17),'open']=1 # working hours
checkin_sim.ix[(checkin_sim.hour==12),'open']=0 # lunch break
checkin_sim.ix[checkin_sim.dayofweek>4,'open']=0 # weekends
checkin_sim.ix[(checkin_sim.month == 8) & (checkin_sim.day<15),'open']=0 # summer holidays. 
checkin_sim.ix[(checkin_sim.month == 1) & (checkin_sim.day<7),'open']=0 # new years.
checkin_sim.ix[(checkin_sim.month == 12) & (checkin_sim.day>24),'open']=0 # x-mass
fft = np.fft.fft(checkin_sim.open)
plt.xlim(10,3000)
plt.ylim(0.01,10**4)
for x in [100, 200, 300, 400]:
    plt.axvline(x,color='red', ls='--')
for x in [699,699*2, 699*3]:
    plt.axvline(x,color='green', ls='--')
for x in [499, 599, 799, 899, 1298]:
    plt.axvline(x,color='orange', ls='--')
for x in [1298]:
    plt.axvline(x,color='black', ls='--')    
plt.loglog(np.sqrt(fft * fft.conj()).real)
plt.xlabel('Number of events')
df.time.max()/(699*24*60)