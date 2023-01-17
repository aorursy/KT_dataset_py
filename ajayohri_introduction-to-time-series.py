
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.core import datetools


df=pd.read_csv('../input/international-airline-passengers.csv')
df.head()

start = datetime.datetime.strptime("1949-01-01", "%Y-%m-%d")
type(start)
print(start)

len(df)

date_list = [start + relativedelta(months=x) for x in range(0,df.shape[0])]
print(date_list[0:4])

for x in range(0,144):
    print(x)


df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None
df.columns
df.columns=('Month','AirPassengers')
#del df['Unnamed: 0']
del df['Month']
df.head()
df.info()
np.count_nonzero(df)
df['AirPassengers'] = df.AirPassengers*1000

#plotting the data
import matplotlib.pyplot as plt

df.AirPassengers.plot(figsize=(12,8), title= 'Monthly Passengers', fontsize=14)
plt.savefig('month_ridership.png', bbox_inches='tight')
#building the model by first decomposing time series in trend and seasonality and irregularity
decomposition = seasonal_decompose(df.AirPassengers, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
print(p)
import itertools
import warnings
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
y=df

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
warnings.filterwarnings("ignore") # specify to ignore warning messages
c3=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            c3.append( results.aic)
        except:
            continue
c3
import numpy as np
index_min = np.argmin(c3)
#minimum AIC for the models
np.min(c3)

index_min

c3[index_min]

warnings.filterwarnings("ignore") # specify to ignore warning messages
c4=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            c4.append('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
c4[index_min]

order1=c4[index_min][6:13]
order1
order1=[int(s) for s in order1.split(',')]
order1

seasonal_order1=c4[index_min][16:27]
seasonal_order1
seasonal_order1=[int(s) for s in seasonal_order1.split(',')]
seasonal_order1
from statsmodels.tsa.x13 import x13_arima_select_order

mod = sm.tsa.statespace.SARIMAX(df.AirPassengers, trend='n', order=order1, seasonal_order=seasonal_order1)

results = mod.fit()
print (results.summary())
df[121:144]

results.predict(start=120,end=144)

type(df)
type(results.predict(start=120,end=144))
y.tail()
df2=pd.DataFrame(results.predict(start=120,end=144))
y.head()
df2.head()
ax = df.plot(figsize=(12,8), title= 'Monthly Passengers', fontsize=14)
df2.plot(ax=ax)