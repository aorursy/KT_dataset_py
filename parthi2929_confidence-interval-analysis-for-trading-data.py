import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
btc=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv")  #importing csv file 
btc.head()
btc = btc.dropna()  # remove NaN they do not help
btc["Timestamp"]=pd.to_datetime(btc["Timestamp"],unit="s") 

hour=btc["Timestamp"]==btc["Timestamp"].dt.floor("H")  # there are over 3 million entries in the dataframe
df=btc[hour]                                    # to make the dataset more simple i only take daily values

df = df[(df['Timestamp'] > '2017-06-20 00:00:00') & (df['Timestamp'] <= '2017-07-23 00:00:00')]
df.head()
df['dX'] = (df['Weighted_Price'].shift(-1) - df['Weighted_Price'])/df['Weighted_Price'].shift(-1)
df['dX'] = df['dX'].shift(1)
# df['dX'] = df['dX'].round(5)  # rounding to 3 decimal places for better frequency distribution later
df = df.dropna()
df.head()
import matplotlib.pyplot as plt
fig,axr = plt.subplots(2,1,figsize=(14,5))

T = df.Timestamp
X = df.Weighted_Price
dX = df.dX

ax = axr[0]
ax.plot(T,X , color="green", label="BTC/USD")      # line plot for seeing the daily weighted price
ax.set_xlabel ("Time")
ax.set_ylabel("USD")

ax = axr[1]
ax.plot(T,dX , color="green", label="BTC/USD")      # line plot for seeing the delta
ax.set_xlabel ("Time")
ax.set_ylabel("USD_Normalized_Delta")

plt.legend() 
plt.show()
# freq = df['dX'].value_counts()
X = df['dX'].tolist()

fig, ax = plt.subplots(1,1, figsize=(7,5))

ax.hist(X, bins=50)

from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.show()
mean = sum(X)/len(X)
var = sum([ (i - mean)**2 for i in X ])/len(X)
from math import sqrt
sd = sqrt(var)
meanstr = str.format('{0:.6f}', mean) # this is a string to print in desired decimal places
sdstr = str.format('{0:.6f}', sd)
print(meanstr, sdstr)
n = 1 # because each sample set size is 1
l_ci, h_ci = mean - 1.96*(sd/sqrt(n)), mean + 1.96*(sd/sqrt(n))
round(l_ci,4), round(h_ci,4)
from IPython.display import HTML
html = '<iframe width="418" height="235" src="https://www.youtube.com/embed/xku0dnLWkcI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
HTML(html)
total_samples = len(X)
n_outliers = total_samples*0.05  # about 5% are expected to fall outside as per the distribution we saw earlier
print(total_samples, n_outliers, (n_outliers/total_samples)*100)
n = 1 # because each sample set size is 1
l_ci, h_ci = mean - 6.5*(sd/sqrt(n)), mean + 6.5*(sd/sqrt(n))
round(l_ci,4), round(h_ci,4)
df['dI'] = 0
# df.loc[df['dX'] > 0.0195, 'dI'] = 1   # using 5% significance as we do not have much of crash in our chosen window
df.loc[df['dX'] < -0.0195, 'dI'] = 1
df.head()
fig,axr = plt.subplots(3,1,figsize=(14,5))

T = df.Timestamp
X = df.Weighted_Price
dX = df.dX
dI = df.dI

ax = axr[0]
ax.plot(T,X , color="green", label="BTC/USD")      # line plot for seeing the daily weighted price
ax.set_xlabel ("Time")
ax.set_ylabel("USD")

ax = axr[1]
ax.plot(T,dX , color="green", label="BTC/USD")      # line plot for seeing the delta
ax.set_xlabel ("Time")
ax.set_ylabel("USD_Normalized_Delta")

ax = axr[2]
ax.plot(T,dI , color="green", label="BTC/USD")      # line plot for seeing the delta
ax.set_xlabel ("Time")
ax.set_ylabel("Indicator")

plt.legend() 
plt.show()