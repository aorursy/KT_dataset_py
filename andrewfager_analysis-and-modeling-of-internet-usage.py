import pandas as pd

import seaborn as sns

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt



plt.rcParams['font.size'] = 14



sns.set_style("ticks")

sns.set_context("paper")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_cdrs = pd.DataFrame({})

for i in range(1,8):

    df = pd.read_csv('../input/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['datetime'])

    df_cdrs = df_cdrs.append(df)

    

df_cdrs=df_cdrs.fillna(0)

df_cdrs['sms'] = df_cdrs['smsin'] + df_cdrs['smsout']

df_cdrs['calls'] = df_cdrs['callin'] + df_cdrs['callout']

df_cdrs.head()



df_cdrs_internet = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()

df_cdrs_internet['hour'] = df_cdrs_internet.datetime.dt.hour+24*(df_cdrs_internet.datetime.dt.day-1)

df_cdrs_internet = df_cdrs_internet.set_index(['hour']).sort_index()



import geojson

import matplotlib.colors as colors

import matplotlib.cm as cmx

#import matplotlib as mpl

from descartes import PolygonPatch



num = int(10000+1)

arr_cellID = np.zeros(num)

arr_mean = np.zeros(num)

for i in range(1,num):

    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']

    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index

    mean = np.mean(ydata)

    arr_cellID[i]=i

    arr_mean[i]=mean



arr_mean[arr_mean<=0] = 1 #replacing 0's with 1's for log calc

arr_mean_log = np.log(arr_mean)



#https://gis.stackexchange.com/questions/93136/how-to-plot-geo-data-using-matplotlib-python

with open("../input/milano-grid.geojson") as json_file:

    json_data = geojson.load(json_file)

    



fig = plt.figure() 

ax = fig.gca() 



coordlist = json_data.features[1]['geometry']['coordinates'][0]



jet = cm = plt.get_cmap('hot') 

#cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_mean))

cNorm  = colors.Normalize(vmin=0, vmax=5000)

#cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_mean_log))

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

#print(scalarMap.get_clim())



for i in range(1,10000):

    poly = json_data.features[i]['geometry']

    colorVal = scalarMap.to_rgba(arr_mean[i])

    ax.add_patch(PolygonPatch(poly, fc=colorVal, ec=colorVal, alpha=1, zorder=1 ))

ax.axis('scaled')



fig.set_size_inches(11,11)

plt.title("Internet connections for each cell in Milan (Black ~ 0 connections, White ~ 5000 connections)")

plt.show()


#Duomo (downtown) data

ydata = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet']

xdata = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].index



f = plt.figure()

plt.plot(xdata, ydata, color='black', linewidth=1, linestyle='-', label='Duomo - data')

plt.title("Internet Connections - Duomo")

plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,168])

plt.ylim([0,10000])

plt.legend()

sns.despine()

plt.show()
# Autocorellation (ACF) calc

from statsmodels.tsa.stattools import acf, pacf

acf_y = acf(ydata, nlags=48)



#Plot ACF: 

f = plt.figure()

plt.plot(acf_y, color='black', linewidth=1, linestyle='-', label='Duomo - data')

plt.axhline(y=0,linestyle='--',color='gray')

plt.title('Autocorrelation Function')

plt.xlabel("Lag [hour]")

plt.ylabel("Autocorrelation [-]")

plt.xlim([0,48])

plt.ylim([-1,1])

plt.legend()

sns.despine()

plt.show()
import scipy

def func(xdata, a,b,c):

    return a*np.sin(2*np.pi*(1/24)*xdata+b)+c
popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)



print(popt)



f = plt.figure()

yfit = func(xdata, *popt)

#residual

residual = ydata - yfit

rss = np.sum(residual**2)

mean = np.mean(ydata)

b = popt[1] #phase shift (b) from curve_fit

#print('rss_navigli',rss_navigli,'mean_navigli',mean_navigli,'rss-norm',rss_navigli/mean_navigli)

#stddev = np.std(residual_navigli)

#print(np.std(residual_navigli))



yfit_duomo = yfit

b_duomo = b

ydata_duomo = ydata

xdata_duomo = xdata

T_peak_duomo = 18-(b_duomo%(2*np.pi))*(24/(2*np.pi))



ydata_moving_avg = ydata.rolling(window=24,center=False).mean()

residual_moving_avg = residual.rolling(window=24,center=False).mean()





f = plt.figure()

plt.plot(xdata, ydata, color='black', linewidth=1, linestyle='--', label='Duomo - data')

plt.plot(xdata, yfit, color='black', linewidth=3, label='Duomo - model')

plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,168])

plt.ylim([0,10000])

plt.legend()

sns.despine()

plt.show()
ff = plt.figure()

plt.plot(xdata, residual, color='black', linewidth=1, linestyle='-', label='Duomo - residual')

plt.axhline(y=0,linestyle='--',color='gray')

plt.title("Residual - Duomo")

plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,168])

plt.ylim([-10000,10000])

plt.legend()

sns.despine()

plt.show()
#Plot ACF: 

from statsmodels.tsa.stattools import acf, pacf

acf_r = acf(residual, nlags=48)



f = plt.figure()



plt.plot(acf_y, color='black', linewidth=1, linestyle='-', label='Duomo - data')

plt.plot(acf_r, color='black', linewidth=1, linestyle='--', label='Duomo - residual')



plt.axhline(y=0,linestyle='--',color='gray')

plt.title('Autocorrelation Function')

plt.xlabel("Lag [hour]")

plt.ylabel("Autocorrelation [-]")

plt.xlim([0,48])

plt.ylim([-1,1])

plt.legend()

sns.despine()

plt.show()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries,maxlag_input=None):

    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)

    #print(dftest)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput) 





test_stationarity(residual,1)
from statsmodels.tsa.stattools import adfuller

def test_stationarity2(timeseries,maxlag_input=None):

    #modifying method - removing prints, return key values

    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

    #Perform Dickey-Fuller test:

    dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    return dftest



num = int(10000+1)

#num = int(10+1)

arr_cellID = np.zeros(num)

arr_mean = np.zeros(num)

arr_rss = np.zeros(num)

arr_rss_norm = np.zeros(num)

arr_test_statistic = np.zeros(num)

arr_critical_value = np.zeros(num)

arr_stationary = np.zeros(num)

arr_T_peak = np.zeros(num)

arr_a_norm = np.zeros(num)





for i in range(1,num):

    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']

    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index

    

    #print(ydata)

    

    #sine fit

    popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)

    

    #

    yfit = func(xdata, *popt)

    

    a=popt[0]

    b=popt[1]

    c=popt[2]

    

    #residual

    residual = ydata - yfit

    rss = np.sum(residual**2)

    mean = np.mean(ydata)

    rss_norm = rss/mean

    #print('rss',rss,'mean',mean,'rss-norm',rss_norm)

    

    #Dickey-Fuller test

    #lag = 1 - using original (not augmented) Dickey-Fuller

    result = test_stationarity2(residual,1)

    test_statistic = result[0]

    critical_value = result[4]['5%']

    stationary = bool(test_statistic<critical_value)



    if (a>0) :

        T_peak = 6+24-(b%(2*np.pi))*(24/(2*np.pi))

    elif (a<=0) :

        T_peak = 18-(b%(2*np.pi))*(24/(2*np.pi))

    T_peak = T_peak%24 



    a_norm = np.abs(a/c)

    

    arr_cellID[i]=i

    arr_mean[i]=mean

    arr_rss[i]=rss

    arr_rss_norm[i]=rss_norm

    arr_test_statistic[i]=test_statistic

    arr_critical_value[i]=critical_value

    arr_stationary[i]=stationary

    arr_T_peak[i]=T_peak

    arr_a_norm[i]=a_norm

    


# Navigli

ydata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet']

xdata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].index





popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)



print(popt)



f = plt.figure()

yfit = func(xdata, *popt)

#residual

residual = ydata - yfit

rss = np.sum(residual**2)

mean = np.mean(ydata)

a = popt[0] #amplitude

b = popt[1] #phase shift (b) from curve_fit

c = popt[2] #mean



yfit_navigli = yfit

b_navigli = b

ydata_navigli = ydata

xdata_navigli = xdata



if (a>0) :

    T_peak = 6+24-(b%(2*np.pi))*(24/(2*np.pi))

elif (a<=0) :

    T_peak = 18-(b%(2*np.pi))*(24/(2*np.pi))

T_peak = T_peak%24 



a_norm = np.abs(a/c)



print (a_norm)



T_peak_navigli = T_peak    

print(T_peak_duomo)

print(T_peak_navigli)
f = plt.figure()

plt.plot(xdata_duomo, ydata_duomo, color='black', linewidth=1, linestyle='--', label='Duomo - data')

plt.plot(xdata_duomo, yfit_duomo, color='black', linewidth=3, label='Duomo - model')

plt.plot(xdata_navigli, ydata_navigli, color='green', linewidth=1, linestyle='--', label='Navigli - data')

plt.plot(xdata_navigli, yfit_navigli, color='green', linewidth=3, label='Navigli - model')



plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,168])

plt.legend()

sns.despine()
f = plt.figure()

#plt.plot(xdata_duomo, ydata_duomo, color='black', linewidth=1, linestyle='--', label='Duomo - data')

plt.plot(xdata_duomo, yfit_duomo, color='black', linewidth=3, label='Duomo - model')

#plt.plot(xdata_navigli, ydata_navigli, color='green', linewidth=1, linestyle='--', label='Navigli - data')

plt.plot(xdata_navigli, yfit_navigli, color='green', linewidth=3, label='Navigli - model')

plt.axvline(x=T_peak_duomo, color='red')

plt.axvline(x=T_peak_navigli, color='red')



plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,24])

plt.ylim([0,10000])

plt.legend()

sns.despine()
fig = plt.figure() 

ax = fig.gca() 



jet = cm = plt.get_cmap('hot') 

cNorm  = colors.Normalize(vmin=12, vmax=20)

#cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_T_peak))

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)



for i in range(1,10000):

    poly = json_data.features[i]['geometry']

    colorVal = scalarMap.to_rgba(arr_T_peak[i])

    ax.add_patch(PolygonPatch(poly, fc=colorVal, ec=colorVal, alpha=1, zorder=2 ))

ax.axis('scaled')

plt.title("Time of day for peak internet usage (Black=noon, White=10PM)")

fig.set_size_inches(11,11)

plt.show()
f = plt.figure()

#plt.plot(xdata_duomo, ydata_duomo, color='black', linewidth=1, linestyle='--', label='Duomo - data')

plt.plot(xdata_duomo, yfit_duomo, color='black', linewidth=3, label='Duomo - model')

#plt.plot(xdata_navigli, ydata_navigli, color='green', linewidth=1, linestyle='--', label='Navigli - data')

plt.plot(xdata_navigli, yfit_navigli, color='green', linewidth=3, label='Navigli - model')



plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,168])

plt.ylim([0,10000])

plt.legend()

sns.despine()
fig = plt.figure() 

ax = fig.gca() 



jet = cm = plt.get_cmap('hot') 

cNorm  = colors.Normalize(vmin=0, vmax=1)

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)



for i in range(1,10000):

    poly = json_data.features[i]['geometry']

    colorVal = scalarMap.to_rgba(arr_a_norm[i])

    ax.add_patch(PolygonPatch(poly, fc=colorVal, ec=colorVal, alpha=1, zorder=1 ))

ax.axis('scaled')

#ax.colorbar(jet, ticks=[0, 1], orientation='horizontal')



#cbar = plt.colorbar(cNorm)

#cbar.ax.set_ylabel('verbosity coefficient')

# Add the contour line levels to the colorbar

#cbar.add_lines(CS2)

plt.title("Ratio of Amplitude to Mean - Amplitude/Mean (Black=0, White=1)")

fig.set_size_inches(11,11)

plt.show()
#Bocconi

ydata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet']

xdata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].index



popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)

print(popt)



yfit = func(xdata, *popt)

residual = ydata - yfit

f = plt.figure()

plt.plot(xdata, ydata, color='blue', linewidth=1, linestyle='--', label='Bocconi - data')

plt.plot(xdata, yfit, color='blue', linewidth=3, label='Bocconi - model')



plt.xlabel("Time [hour]")

plt.ylabel("Internet Connections [#]")

plt.xlim([0,168])

plt.ylim([0,10000])

plt.legend()

sns.despine()



f = plt.figure()

plt.plot(xdata, residual, color='blue', linewidth=1, linestyle='-', label='Bocconi - residual')

plt.axhline(y=0,linestyle='--',color='gray')

plt.title("Residual - Bocconi")

plt.xlabel("Time [hour]")

plt.ylabel("Residual Internet Connections [#]")

plt.xlim([0,168])

plt.ylim([-10000,10000])

plt.legend()

sns.despine()
test_stationarity(residual,1)
f = plt.figure()

plt.plot(arr_cellID, arr_test_statistic,'ko')

plt.axhline(y=arr_critical_value[-1], linewidth=2, color = 'red', label='DF Critical Value (95%)')

plt.legend()

plt.xlabel("Cell ID []")

plt.ylabel("DF Test Statistic")

sns.despine()