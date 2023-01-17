import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
%matplotlib inline 
import scipy.integrate as integrate
from scipy.optimize import curve_fit
pd.options.display.max_rows = 12
data = pd.read_csv("../input/data.txt", sep = ' ',header = None, names = ['date', 'time','epoch','moteid','temp','humidity','light','voltage'])
data.head()
data.shape
data.describe()
data.isnull().sum()
data['temp'].plot()
data.fillna(0, inplace=True)
data['epoch'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
data['epoch'] = data['epoch'].astype(int)
data.isnull().sum()
data.groupby('moteid').mean()
data['Timestamp'] = data[['date', 'time']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
new_data = data
data.drop(['date','time'],axis=1,inplace =True)
data.set_index(pd.to_datetime(data.Timestamp), inplace=True)
data[['moteid','temp','humidity','light','voltage']] = data[['moteid','temp','humidity','light','voltage']].apply(pd.to_numeric)
data['moteid'].value_counts()
moteid_grp = data.groupby(['moteid'])
corr_id = moteid_grp.corr(method='pearson')
corr_id.fillna(0, inplace=True)
corr_id
data.head()
new = data['2004/3/1':'2004/3/21']

#I create two datafranes to hold the data used for covariance calculations
Temperature_data = pd.DataFrame()
Humidity_data = pd.DataFrame()
test = pd.DataFrame()

#Write data from Node 14
test = new.loc[data.moteid==14]
test = test.groupby([test.index.year,test.index.month,test.index.day, test.index.hour]).mean()
Temperature_data [['Node14']] = test[['temp']]
Humidity_data[['Node14']] = test[['humidity']]

#Write data from Nodes 22 to 29
for i in range(8):
    j = i + 22
    test = new.loc[data.moteid==j]
    test = test.groupby([test.index.year,test.index.month,test.index.day, test.index.hour]).mean()
    Temperature_data[['Node' + str(j)]] = test[['temp']]
    Humidity_data[['Node'+ str(j)]] = test[['humidity']]
Temperature_data
Humidity_data
node_distance = pd.read_csv('../input/intel_nodes_distances.csv')

lists_hy = []
lists_hx = []
    
for z in range(10):
    # Calculation of the covariance for lag i hour for temp
    humy = [None] *45
    humx = [None] *45
    k=0
    i=0
    for first_column in Temperature_data:
        df1 = Temperature_data[first_column][:-(1+z)]
        std_test1=np.std(df1)
        j = 0 
        for second_column in Temperature_data:
            if j <= i:
                df2 = Temperature_data[second_column][(1+z):]
                std_test2=np.std(df2)
                humy[k] = (np.cov(df1,df2)/(std_test1*std_test2)).item((0, 1))
                humx[k] = node_distance.iloc[i,j]
                j = j + 1
                k = k + 1
        i = i + 1
    lists_hy.append(humy)
    lists_hx.append(humx)
temperature_timex = []
temperature_timey = []
for i in range(10):
    average = (sum(lists_hy[i])/len(lists_hy[i]))
    temperature_timey.append(average)

for z in range(10):
    temperature_timex.append(z)

plt.scatter(temperature_timex, temperature_timey)
from statsmodels.tsa.stattools import acf,pacf
lag_acf = acf(data['temp'])
#Plot pACF: a
plt.subplot(121) 
plt.plot(lag_acf)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(60,20))

for xcol, ax in zip(['humidity', 'temp', 'light','voltage'], axes):
    data.plot(kind='scatter', x='epoch', y=xcol, ax=ax, alpha=1, color='r')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,30))
for xcol, ax in zip(['temp', 'light','humidity'], axes):
    data.plot(kind='scatter', x='voltage', y=xcol, ax=ax, color='b')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,30))
for xcol, ax in zip(['temp', 'humidity','voltage'], axes):
    data.plot(kind='scatter', x='light', y=xcol, ax=ax, alpha=1, color='g')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,30))
for xcol, ax in zip(['temp', 'light','voltage'], axes):
    data.plot(kind='scatter', x='humidity', y=xcol, ax=ax, alpha=1, color='y')
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(60,30))
for xcol, ax in zip(['humidity', 'light','voltage'], axes):
    data.plot(kind='scatter', x='temp', y=xcol, ax=ax, alpha=1, color='b')
data.index
index_hour = data.index.hour
df1_by_hour =data['temp'].groupby(index_hour).mean()
df1_by_hour.plot()
plt.show()
data.index
index_hour = data.index.hour
df1_by_hour =data['humidity'].groupby(index_hour).mean()
df1_by_hour.plot()
plt.show()
data.index
index_hour = data.index.hour
df1_by_hour =data['light'].groupby(index_hour).mean()
df1_by_hour.plot()
plt.show()
data.index
index_hour = data.index.hour
df1_by_hour =data['voltage'].groupby(index_hour).mean()
df1_by_hour.plot()
plt.show()
new_data.plot(subplots=True,linewidth=0.5,
                layout=(2, 4),figsize=(60, 20),
                sharex=False,
                sharey=False)

plt.show()
new_data.corr(method='pearson')
from matplotlib import pyplot as plt 
d_m21 = data.loc[data['moteid'] == 21.0]
d_m22 = data.loc[data['moteid'] == 22.0]
d_m10 = data.loc[data['moteid'] == 10.0]
fig2 = plt.figure(figsize = (15,10))
d_m21['temp'].plot(label='temperature for moteid=21.0')
d_m22['temp'].plot(label='temperature for moteid=22.0')
fig2.suptitle('Variation in temperature over time for moteid= 21.0 and 22.0', fontsize=10)
plt.xlabel('timestamp', fontsize=10)
plt.ylabel('temperature', fontsize=10)
plt.legend()
from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
%matplotlib inline
def mov_average(data, window_size):

    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')
def find_anomalies(y, window_size, sigma=1.0):
    avg = mov_average(y, window_size).tolist()
    residual = y - avg
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(count(), y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}
def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
   
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = mov_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(0, 40000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)
    events = {}
    events = find_anomalies(y, window_size=window_size, sigma=sigma_value)
    

    x_anom = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anom = np.fromiter(events['anomalies_dict'].values(), dtype=float,count=len(events['anomalies_dict']))
    plt.plot(x_anom, y_anom, "b*")
    print(x_anom)
    plt.grid(True)
    plt.show()
x = d_m10['epoch']
Y = d_m10['temp']
plot_results(x, y=Y, window_size=50, text_xlabel="Date", sigma_value=3,text_ylabel="temperature")








