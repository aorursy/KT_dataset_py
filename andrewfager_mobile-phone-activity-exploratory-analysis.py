import pandas as pd

import seaborn as sns

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt



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

#print(df_cdrs_internet)
f = plt.figure()



ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(ax=ax, label='Navigli')

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()



# Shrink current axis's height by 10% on the bottom

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1,

                 box.width, box.height * 0.9])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),

          fancybox=True, shadow=True, ncol=5)
boxplots = {

    'calls': "Calls",

    'sms': "SMS",

    "internet": "Internet CDRs"

}



df_cdrs_internet['weekday'] = df_cdrs_internet.datetime.dt.weekday



f, axs = plt.subplots(len(boxplots.keys()), sharex=True, sharey=False)

f.subplots_adjust(hspace=.35,wspace=0.1)

i = 0

plt.suptitle("")

for k,v in boxplots.items():

    ax = df_cdrs_internet.reset_index().boxplot(column=k, by='weekday', grid=False, sym='', ax =axs[i])

    axs[i].set_title(v)

    axs[i].set_xlabel("")

    sns.despine()

    i += 1

    

plt.xlabel("Weekday (0=Monday, 6=Sunday)")

f.text(0, 0.5, "Number of events", rotation="vertical", va="center")
ydata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet']

xdata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].index



import scipy



def func(xdata, a,b,c,d):

    return a*np.sin(b*xdata+c)+d



def func2(xdata, a,b,c):

    return a*np.sin(2*np.pi*(1/24)*xdata+b)+c



popt,pcov = scipy.optimize.curve_fit(func2, xdata, ydata)



print(popt)



f = plt.figure()

yfit = func2(xdata, *popt)

#residual

residual_navigli = ydata - yfit

rss_navigli = np.sum(residual_navigli**2)

mean_navigli = np.mean(ydata)

print('rss_navigli',rss_navigli,'mean_navigli',mean_navigli,'rss-norm',rss_navigli/mean_navigli)

stddev = np.std(residual_navigli)

#print(np.std(residual_navigli))



#ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

#df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

ax = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(label='Navigli')

plt.plot(xdata, ydata)

plt.plot(xdata, yfit)

plt.plot(xdata, yfit+stddev)

plt.plot(xdata, yfit-stddev)



plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()







f = plt.figure()

plt.plot(xdata, residual_navigli)



plt.xlabel("Weekly hour")

plt.ylabel("Residual")

sns.despine()



#Bocconi

ydata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet']

xdata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].index





popt,pcov = scipy.optimize.curve_fit(func2, xdata, ydata)



print(popt)

yfit = func2(xdata, *popt)





f = plt.figure()



#ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

#ax = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

#ax = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(label='Navigli')

plt.plot(xdata, ydata, label='Bocconi')

plt.plot(xdata, yfit, label='Fit')



plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()



#residual

residual_bocconi = ydata - yfit

rss = np.sum(residual_bocconi**2)

mean = np.mean(ydata)

print('rss_',rss,'mean',mean,'rss-norm',rss/mean)



f = plt.figure()

plt.plot(xdata, residual_bocconi)



plt.xlabel("Weekly hour")

plt.ylabel("Residual")

sns.despine()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries,maxlag_input=None):

    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

    #Determing rolling statistics

    #rolmean = pd.rolling_mean(timeseries, window=12)

    #rolstd = pd.rolling_std(timeseries, window=12)

    #ydata_moving_avg = ydata.rolling(window=24,center=False).mean()

    #ydata_moving_stddev = ydata.rolling(window=24,center=False).std()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)

    #print(dftest)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput) 



#ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

#df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

#df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(ax=ax, label='Navigli')    



#print('\nOriginal-Duomo')

#ydata_duomo = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet']

#test_stationarity(ydata_duomo,24)



print('\nResidual-Bocconi')

test_stationarity(residual_bocconi,1)



print('\nResidual-Navigli')

test_stationarity(residual_navigli,1)




from statsmodels.tsa.stattools import adfuller

def test_stationarity2(timeseries,maxlag_input=None):

    #modifying method - removing prints, return key values

    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/



    

    #Perform Dickey-Fuller test:

    #print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)

    #print(dftest)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    #print(dfoutput) 

    return dftest





num = int(10000+1)

arr_cellID = np.zeros(num)

arr_mean = np.zeros(num)

arr_rss = np.zeros(num)

arr_rss_norm = np.zeros(num)

arr_test_statistic = np.zeros(num)

arr_critical_value = np.zeros(num)

arr_stationary = np.zeros(num)





for i in range(1,num):

    #ydata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet']

    #xdata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].index

    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']

    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index

    

    #print(ydata)

    

    #sine fit

    popt,pcov = scipy.optimize.curve_fit(func2, xdata, ydata)

    

    #

    yfit = func2(xdata, *popt)

    

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



    arr_cellID[i]=i

    arr_mean[i]=mean

    arr_rss[i]=rss

    arr_rss_norm[i]=rss_norm

    arr_test_statistic[i]=test_statistic

    arr_critical_value[i]=critical_value

    arr_stationary[i]=stationary

    

#print(result)

#print('test_statistic',test_statistic)

#print('critical_value',critical_value)

#print('Stationary?',bool(test_statistic<critical_value))

f = plt.figure()

plt.plot(xdata, ydata, label='Ydata')

plt.plot(xdata, yfit, label='Fit')

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()



f = plt.figure()

plt.plot(xdata, residual)

plt.xlabel("Weekly hour")

plt.ylabel("Residual")

sns.despine()



f = plt.figure()

plt.plot(arr_cellID, arr_mean,'bo')

#plt.hist(cell['internet'],100)

plt.xlabel("Cell ID []")

plt.ylabel("mean")

sns.despine()



f = plt.figure()

plt.plot(arr_cellID, arr_rss,'ro')

#plt.hist(cell['internet'],100)

plt.xlabel("Cell ID []")

plt.ylabel("RSS")

sns.despine()



f = plt.figure()

plt.plot(arr_cellID, arr_rss_norm,'ro')

#plt.hist(cell['internet'],100)

plt.xlabel("Cell ID []")

plt.ylabel("RSS-norm")

sns.despine()



f = plt.figure()

plt.plot(arr_cellID, arr_test_statistic,'ro')

plt.axhline(y=arr_critical_value[-1], linewidth=2, color = 'k')

#plt.hist(cell['internet'],100)

plt.xlabel("Cell ID []")

plt.ylabel("test statistic")

sns.despine()



#    arr_cellID[i]=i

#    arr_mean[i]=mean

#    arr_rss[i]=rss

#    arr_rss_norm[i]=rss_norm

#    arr_test_statistic[i]=test_statistic

#    arr_critical_value[i]=critical_value

#    arr_stationary[i]=stationary
from scipy import signal

def func_square(t, a,b,c):

    return a*signal.square(2 * np.pi * b * t)+c



ydata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet']

xdata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].index





popt,pcov = scipy.optimize.curve_fit(func_square, xdata, ydata, p0=[1000,0.05,4000])



#print(popt)

yfit = func_square(xdata, *popt)



f = plt.figure()

plt.plot(xdata, ydata, label='Navigli')

plt.plot(xdata, yfit, label='Fit')

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

#moving_avg = pd.rolling_mean(ts_log,12)

ydata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet']

#ydata_moving_avg = pd.rolling_mean(ydata,24) #depracated, use below

ydata_moving_avg = ydata.rolling(window=24,center=False).mean()

ydata_moving_stddev = ydata.rolling(window=24,center=False).std()





#ts_log_moving_avg_diff = ts_log - moving_avg

#ts_log_moving_avg_diff.head(12)

ydata_diff = ydata - ydata_moving_avg





f = plt.figure()

plt.plot(xdata, ydata, label='Navigli')

plt.plot(xdata, ydata_moving_avg, label='moving-avg')

plt.plot(xdata, ydata_moving_stddev, label='moving-stddev')

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()



f = plt.figure()

plt.plot(xdata, ydata, label='Navigli')

plt.plot(xdata, ydata_diff, label='Navigli')

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries,maxlag_input=None):

    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

    #Determing rolling statistics

    #rolmean = pd.rolling_mean(timeseries, window=12)

    #rolstd = pd.rolling_std(timeseries, window=12)

    #ydata_moving_avg = ydata.rolling(window=24,center=False).mean()

    #ydata_moving_stddev = ydata.rolling(window=24,center=False).std()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)

    #print(dftest)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput) 



#ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

#df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

#df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(ax=ax, label='Navigli')    



print('\nOriginal-Duomo')

ydata_duomo = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet']

test_stationarity(ydata_duomo,24)



print('\nOriginal-Bocconi')

ydata_bocconi = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet']

test_stationarity(ydata_bocconi,24)



print('\nOriginal-Navigli')

ydata_navigli = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet']

test_stationarity(ydata_navigli,24)



#ydata_diff has trouble converging - not sure why??

#print('Moving-avg differenced')

#test_stationarity(ydata_diff)
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf



#############################

print('Navigli: ACF & PACF')

navigli_acf = acf(ydata_navigli, nlags=48)

navigli_pacf = pacf(ydata_navigli, nlags=48, method='ols')



#Plot ACF: 

plt.subplot(121) 

plt.plot(navigli_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ydata_navigli)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ydata_navigli)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(navigli_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ydata_navigli)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ydata_navigli)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()



#############################

print('Bocconi: ACF & PACF')

bocconi_acf = acf(ydata_bocconi, nlags=48)

bocconi_pacf = pacf(ydata_bocconi, nlags=48, method='ols')



#Plot ACF: 

plt.subplot(121) 

plt.plot(bocconi_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ydata_bocconi)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ydata_bocconi)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(bocconi_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ydata_bocconi)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ydata_bocconi)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
df_cdrs_internet2 = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()

#df_cdrs_internet2.head()

df_cdrs_internet2['hour'] = df_cdrs_internet2.datetime.dt.hour+24*(df_cdrs_internet2.datetime.dt.day-1)

#df_cdrs_internet2.head()

df_cdrs_internet2 = df_cdrs_internet2.set_index(['datetime']).sort_index()

df_cdrs_internet2.head()



ts =df_cdrs_internet2[df_cdrs_internet2.CellID==4456]['internet']

#ts =df_cdrs_internet2[df_cdrs_internet2=4456]['internet']

ts.head()
from statsmodels.tsa.seasonal import seasonal_decompose







#print(len(ydata_navigli))

#ydata_navigli.dropna(inplace=True)

#print(len(ydata_navigli))

#rng = pd.date_range('1/1/2011', periods=168, freq='H')

#ts = pd.Series(ydata_navigli, index=rng)

#ts = ydata_navigli.reindex(rng)

#ydata_navigli.head()

#ts.head()



#do not understand why freq is required. Do not understand what frequency is actually.

# I thought ts is appropriate pandas series w/ timeseries index...

decomposition = seasonal_decompose(ts,model='additive') 



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid







plt.subplot(411)

plt.plot(ts, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()


ts = df_cdrs_internet2.groupby(['hour'], as_index=False).mean()

std = df_cdrs_internet2.groupby(['hour'], as_index=False).std()



ts2 = df_cdrs_internet2[df_cdrs_internet2.CellID==4456]['internet']



cell = df_cdrs_internet2.groupby(['CellID'], as_index=False).mean()



#print(cell)

f = plt.figure()

#plt.plot(ts['internet'], label='avg')

#plt.plot(ts2, label='Navigli')

plt.hist(cell['internet'],100)

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()



f = plt.figure()

plt.plot(ts['internet'], label='avg')

plt.plot(std['internet'], label='std')

#plt.plot(ts['internet']-std['internet'], label='std')

#ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

#df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

#df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(ax=ax, label='Navigli')

#plt.plot(ts2, label='Navigli')

#plt.hist(cell['internet'],100)

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()
import geojson

from descartes import PolygonPatch

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import numpy as np



with open("../input/milano-grid.geojson") as json_file:

    json_data = geojson.load(json_file)

    

#plt.clf()

#ax = plt.figure(figsize=(10,10)).add_subplot(111)#fig.gca()



#m = Basemap(projection='robin', lon_0=0,resolution='c')

#m.drawmapboundary(fill_color='white', zorder=-1)

#m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize=14)

#m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5',fontsize=14)

#m.drawcoastlines(color='0.6', linewidth=1)



#print(json_data.features)



print(json_data.keys())

json_data['crs']



len(json_data.features)



#for i in range(2799):

#    coordlist = json_data.features[i]['geometry']['coordinates'][0]

#    if i < 2796:

#        name = json_data.features[i]['properties']['CTRYNAME']

#        aez =  json_data.features[i]['properties']['AEZ']

#

#    for j in range(len(coordlist)):

#        for k in range(len(coordlist[j])):

#            coordlist[j][k][0],coordlist[j][k][1]=m(coordlist[j][k][0],coordlist[j][k][1])



#    poly = {"type":"Polygon","coordinates":coordlist}#coordlist

#    ax.add_patch(PolygonPatch(poly, fc=[0,0.5,0], ec=[0,0.3,0], zorder=0.2 ))



#ax.axis('scaled')

#plt.draw()

#plt.show()



#https://gis.stackexchange.com/questions/93136/how-to-plot-geo-data-using-matplotlib-python

import matplotlib.pyplot as plt 

from descartes import PolygonPatch



BLUE = '#6699cc'

fig = plt.figure() 

ax = fig.gca() 



coordlist = json_data.features[1]['geometry']['coordinates'][0]



print(json_data.features[1]['geometry'])



#for j in range(len(coordlist)):

#    for k in range(len(coordlist[j])):

#          coordlist[j][k][0],coordlist[j][k][1]=m(coordlist[j][k][0],coordlist[j][k][1])





#poly = {"type":"Polygon","coordinates":coordlist}#coordlist

#ax.add_patch(PolygonPatch(poly, fc=[0,0.5,0], ec=[0,0.3,0], zorder=0.2 ))



import matplotlib.colors as colors

import matplotlib.cm as cmx

import matplotlib as mpl



arr_mean_log = np.log(arr_mean)



jet = cm = plt.get_cmap('jet') 

#cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_mean))

#cNorm  = colors.Normalize(vmin=0, vmax=1000)

cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_mean_log))

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

print(scalarMap.get_clim())



for i in range(1,10000):

    poly = json_data.features[i]['geometry']

    colorVal = scalarMap.to_rgba(arr_mean_log[i])

    ax.add_patch(PolygonPatch(poly, fc=colorVal, ec=colorVal, alpha=0.5, zorder=2 ))

ax.axis('scaled')



#cbar = ax.colorbar(jet, ticks=[0, 1000], orientation='horizontal')

#cbar.ax.set_xticklabels(['Low', 'High'])  # horizontal colorbar



#cmap = mpl.cm.cool

#norm = mpl.colors.Normalize(vmin=5, vmax=10)



# ColorbarBase derives from ScalarMappable and puts a colorbar

# in a specified axes, so it has everything needed for a

# standalone colorbar.  There are many more kwargs, but the

# following gives a basic continuous colorbar with ticks

# and labels.

#cb1 = mpl.colorbar.ColorbarBase(ax, cmap=jet,

#                                norm=cNorm,

#                                orientation='horizontal')

#cb1.set_label('Some Units')



plt.show()