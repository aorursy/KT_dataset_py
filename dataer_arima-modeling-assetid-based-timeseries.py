# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import kagglegym

import random

from sklearn import ensemble, linear_model, metrics

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA 

import dateutil.parser as parser

from statsmodels import tsa



import seaborn as sns

%matplotlib inline

plt.style.use('classic')



sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
env = kagglegym.make()

o = env.reset()

train = o.train  #train dataset - the first half of the full dataframe

print(train.shape) #print shape
#Courtesy Jeff Moser https://www.kaggle.com/jeffmoser/two-sigma-financial-modeling/kagglegym-api-overview

with pd.HDFStore("../input/train.h5", "r") as train1:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train1.get("train") #df is the complete dataframe
def findMatchedColumnsUsingPrefix(prefix, df):

    columns = df.columns[df.columns.str.startswith(prefix)]

    return list(columns.values)
derived_columns = findMatchedColumnsUsingPrefix("derived", df)

fundamental_columns = findMatchedColumnsUsingPrefix("fundamental", df)

technical_columns = findMatchedColumnsUsingPrefix("technical", df)



print("There are {} derived columns".format(len(derived_columns)))

print("There are {} fundamental columns".format(len(fundamental_columns)))

print("There are {} technical columns".format(len(technical_columns)))
#Thanks to Chase:

#https://www.kaggle.com/chaseos/two-sigma-financial-modeling/understanding-id-and-timestamp

# id counts w.r.t time

temp = train.groupby('timestamp').apply(lambda x: x['id'].nunique())#Also can use count() 

len(train)

#as we know the

#id is unique for a timstamp

plt.figure(figsize=(8,4))

plt.plot(temp, color="red")

plt.xlabel('timestamp')

plt.ylabel('id count')

plt.title('Number of ids over time')
# lifespan of each id

temp = train.groupby('id').apply(len)

temp = temp.sort_values()

temp = temp.reset_index()

plt.figure(figsize=(8,4))

plt.plot(temp[0], color="blue")

plt.xlabel('index for each id sorted by number of timestamps')

plt.ylabel('number of timestamps')

plt.title('Number of timestamps ("Lifespan") for each id')

print(temp[0].describe())
byTS=train.pivot(index='timestamp', columns='id', values='y')

byTS.fillna(0,inplace=True)

byTS

#Reset index so that 'timestamp' is a column 

byTS.reset_index(level=0,inplace=True)

byTS.timestamp
datestart = '2010-01-01'

dateend = '2013-06-21'

timeperiods = len(byTS)





dayse=pd.date_range(datestart, freq='B', periods=timeperiods)

dayse
byTS.datetime = pd.to_datetime(dayse,unit='B',errors='coerce')

byTS
TSdict={}



#history = [x for x in train]

for pos,col in enumerate(byTS.columns[1:]):

    ps = pd.Series(byTS[col].values,index=byTS.datetime)

    #TS.append(ps)

     

    TSdict[int(col)] = ps

TSdict

TSdict[25]
from statsmodels.tsa import stattools as stt 

def is_stationary(df, maxlag=15, autolag=None, regression='ct'): 

    """Run the Augmented Dickey-Fuller test from Statsmodels 

    and print output. 

    """ 

    outpt = stt.adfuller(df,maxlag=maxlag, autolag=autolag, 

                         regression=regression) 

    print('adf\t\t {0:.3f}'.format(outpt[0])) 

    print('p\t\t {0:.3g}'.format(outpt[1])) 

    print('crit. val.\t 1%: {0:.3f}, 5%: {1:.3f}, 10%: {2:.3f}'.format(outpt[4]["1%"], outpt[4]["5%"], outpt[4]["10%"])) 

    print('stationary?\t {0}'.format(['true', 'false'][outpt[0]>outpt[4]['5%']])) 

    return outpt 
TS=TSdict[25]

diff1=is_stationary(TS.diff(1).dropna())        



diff1
diff12=is_stationary(TS.diff(1).diff(12).dropna());

diff12
TSdict.keys()
from statsmodels.tsa.seasonal import seasonal_decompose

def decomp(TS):

    TS_decomp = seasonal_decompose(TS, freq=252)

    TS_trend = TS_decomp.trend 

    TS_seasonal = TS_decomp.seasonal 

    TS_residual = TS_decomp.resid

    ts = TS-TS_seasonal 

    tsdiff = ts.diff(1)

    return ts
def despine(axs):

    # to be able to handle subplot grids

    # it assumes the input is a list of 

    # axes instances, if it is not a list, 

    # it puts it in one

    if type(axs) != type([]):

        axs = [axs]

    for ax in axs:

        ax.yaxis.set_ticks_position('left')

        ax.xaxis.set_ticks_position('bottom')

        ax.spines['bottom'].set_position(('outward', 10))

        ax.spines['left'].set_position(('outward', 10))
import seaborn as sns

ids_to_use=[557,1113,1136,2134]

fig = plt.figure(figsize=(8, 20))

plot_count = 0

for id in ids_to_use:

    TS = TSdict[id]

    plot_count += 1

    plt.subplot(4, 1, plot_count)

    title_str = "Differencing for Asset " + str(id) + ':Stationary for 1 Period: {0} '.format(['true', 'false'][diff1[0]>diff1[4]['5%']]) 

   

    title_str = title_str + ', and 12 Periods : {0} '.format(['true', 'false'][diff12[0]>diff12[4]['5%']])

    TS.diff(1).plot(label='1 period', title=title_str, 

                      dashes=(15,5)) 

    TS.diff(1).diff(12).plot(label='1 and 12 period(s)', 

                               color='Coral') 

    plt.legend(loc='best') 

    despine(plt.gca()) 

    plt.xlabel('Date') 

plt.show()
import dateutil.parser as parser

from datetime import *; from dateutil.relativedelta import *

dayse

#Great discovery: Finally figured how to convert

#pd.datatime to str of date

str(TS.index[261])

#begin =parser.parse(dayse[261]).year

#end=parser.parse(dateend).year

#dateend

def change_plot(ax):

    despine(ax)

    ax.locator_params(axis='y', nbins=5)

    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
from random import *

from random import randint

toPlot= sample(list(TSdict.keys()),4)

plot_count=0

for i in toPlot:

    ts1=TSdict[i]

    plot_count += 1

    plt.figure(figsize=(9,4.5))

    plt.subplot(4, 1, plot_count)

    fig=plt.gcf()

    plt.title( 'Y Values for ID: ' + str(i))

    #plt.legend(loc='best')  

    plt.plot(ts1)

plt.show()
#plot forecast for model

from statsmodels.tsa.seasonal import seasonal_decompose 

    

#ids_to_use=[557,1113,1136,2134]

ids_to_use=[25,568,852,1335,1872,2145]

#fig = plt.figure(figsize=(8, 20))

plot_count = 0

for id in ids_to_use:

    TS = TSdict[id]

    #Tried different seasonalities of 5, 22, 66,132,264/260

    TS_decomp = seasonal_decompose(TS, freq=264)

    #TS_decomp = seasonal_decompose(TS, freq=5)

    TS_trend = TS_decomp.trend 

    TS_seasonal = TS_decomp.seasonal 

    TS_residual = TS_decomp.resid

    plt.figure(figsize=(7,4.5))

    fig=plt.gcf()

    fig.suptitle("Decomposition for Asset : " + str(id) + "(seasonality 264 days)", fontsize=12)

    plt.subplot(221)

    plt.plot(TS, color='Green')

    change_plot(plt.gca())

    plt.title('Y Values', color='Green')

    xl = plt.xlim()

    yl = plt.ylim(-0.10,0.10)

    

    plt.subplot(222)

    plt.plot(TS.index,TS_trend, 

         color='Coral')

    change_plot(plt.gca())

    plt.title('Trend', color='Coral')

    plt.gca().yaxis.tick_right()

    plt.gca().yaxis.set_label_position("right")

    plt.xlim(xl)

    plt.ylim(yl)



    plt.subplot(223)

    plt.plot(TS.index,TS_seasonal, 

         color='SteelBlue')

    change_plot(plt.gca())

    plt.gca().xaxis.tick_top()

    plt.gca().xaxis.set_major_formatter(plt.NullFormatter())

    plt.xlabel('Seasonality', color='SteelBlue', labelpad=-20)

    plt.xlim(xl)

    plt.ylim((-0.1,0.1))



    plt.subplot(224)

    plt.plot(TS.index,TS_residual,

        color='IndianRed')

    change_plot(plt.gca())

    plt.xlim(xl)

    plt.gca().yaxis.tick_right()

    plt.gca().yaxis.set_label_position("right")

    plt.gca().xaxis.tick_top()

    plt.gca().xaxis.set_major_formatter(plt.NullFormatter())

    plt.ylim((-0.1,0.1))

    plt.xlabel('Residuals', color='IndianRed', labelpad=-20)

    plt.tight_layout()

    plt.subplots_adjust(hspace=0.55)

    plt.subplots_adjust(top=0.85)

    '''plot_count += 1

    lax=plt.subplot(4, 1, plot_count)

    lax.set_title("Timeseries ID: " + str(id))

    ts = decomp(TS)

    model = ARIMA(ts, order=(1, 1, 0)) 

    #Look for the first element of next year

    arres = model.fit()

    arres.plot_predict(start=str(TS.index[523]), end=dateend, alpha=0.10) 

    plt.legend(loc='upper left') 

    despine(plt.gca()) 

    plt.xlabel('Year') 

    print(arres.aic, arres.bic)

    '''



    
import matplotlib.dates as mpldates

ids_to_use=[2134,1113,1136,557]

#fig = plt.figure(figsize=(8, 20))

plot_count = 0

for id in ids_to_use:

    TS = TSdict[id]

    fig = plt.figure(figsize=(5,1.5) )

    

    ax1 = fig.add_axes([0.1,0.1,0.6,0.9])

    ax1.plot(TS-TS_trend, 

         color='Green', label='Detrended data')

    ax1.plot(TS_seasonal, 

         color='Coral', label='Seasonal component')

    kwrds=dict(lw=1.5, color='0.6', alpha=0.8)

    d1 = pd.datetime(2011,1,3)

    dd = pd.Timedelta('264 Days')

    [ax1.axvline(d1+dd*i, dashes=(3,5),**kwrds) for i in range(4)]

    d2 = pd.datetime(2010,5,1)

    [ax1.axvline(d2+dd*i, dashes=(2,2),**kwrds) for i in range(4)]

    ax1.set_ylim((-0.2,0.2))



    ax1.locator_params(axis='y', nbins=4)

    ax1.set_xlabel('Year')

    ax1.set_title('Y Seasonality for Asset : ' + str(id))

    ax1.set_ylabel('Y')

    ax1.legend(loc=0, ncol=2, frameon=True);



    ax2 = fig.add_axes([0.8,0.1,0.4,0.9])

    ax2.plot(TS_seasonal['2010':'2010'], 

         color='Coral', label='Seasonal component')

    ax2.set_ylim((-0.2,0.2))

    [ax2.axvline(d1+dd*i, dashes=(3,5),**kwrds) for i in range(1)]

    d2 = pd.datetime(2010,5,1)

    [ax2.axvline(d2+dd*i, dashes=(2,2),**kwrds) for i in range(1)]

    despine([ax1, ax2])

    

    yrsfmt = mpldates.DateFormatter('%b')

    ax2.xaxis.set_major_formatter(yrsfmt)

    labels = ax2.get_xticklabels()

    plt.setp(labels, rotation=90);

    

plt.show()
#for id 557

TS_seasonal_component = TS_seasonal['2010'].values

TS_residual.dropna(inplace=True)

is_stationary(TS_residual);
import scipy.stats as st

loc, shape = st.norm.fit(TS_residual)

print(shape)
import scipy.stats as st

axes=plt.gca()

loc, shape = st.norm.fit(TS_residual)

axes.set_ylim([0,52])

axes.set_xlim([-0.035,0.035])

#x=range(-0.8,0.8)

x=0.065

y = st.norm.pdf(x, loc, shape)

n, bins, patches = plt.hist(TS_residual, bins=20, normed=True)

plt.plot(x,y, color='Coral')

despine(axes)

plt.title('Residuals for ID : ' + str(557))

plt.xlabel('Value'); 

plt.ylabel('Counts');
TS.diff(1).plot(label='1 period', title='Y Values for Asset ID : ' + str(557))

plt.legend(loc='best')

despine(plt.gca())
is_stationary(TS.diff(1).dropna());
title_str = "Differencing for Asset " + str(557) + ':Stationary for 1 Period: {0} '.format(['true', 'false'][diff1[0]>diff1[4]['5%']]) 

   

title_str = title_str + ', and 12 Periods : {0} '.format(['true', 'false'][diff12[0]>diff12[4]['5%']])



TS.diff(1).plot(label='1 period', title=title_str,

                      dashes=(15,5))

TS.diff(1).diff(12).plot(label='1 and 12 period(s)',

                               color='Coral')

plt.legend(loc='best')

despine(plt.gca())

plt.xlabel('Date')
is_stationary(TS.diff(1).diff(12).dropna());
is_stationary((TS-TS_seasonal).diff(1).dropna());
ts =TS-TS_seasonal

tsdiff = ts.diff(1)

plt.plot(tsdiff)
model = ARIMA(ts, order=(1, 1, 0))  

arres = model.fit()
#Predict any number of days beyond 905 which is the last period in the train set

endd=str(TS.index[905]+10)

arres.plot_predict(start=str(TS.index[532]), end=endd, alpha=0.10)

plt.legend(loc='upper left')

print(arres.aic, arres.bic)
model = ARIMA(ts, order=(0, 1, 1))  

mares = model.fit() 

mares.plot_predict(str(TS.index[532]), end=endd, alpha=0.10)

plt.legend(loc='upper left');

print(mares.aic, mares.bic)
tsa.stattools.arma_order_select_ic(tsdiff.dropna(), max_ar=3, max_ma=3, ic='aic')

acf = stt.acf(tsdiff.dropna(), nlags=10)

pacf = stt.pacf(tsdiff.dropna(), nlags=10)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,2))

ax1.axhline(y=0,color='gray')

ax1.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')

ax1.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')

ax1.axvline(x=1,ls=':',color='gray')

ax1.plot(acf)

ax1.set_title('ACF')



ax2.axhline(y=0,color='gray')

ax2.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')

ax2.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')

ax2.axvline(x=1,ls=':',color='gray')

ax2.plot(pacf)

ax2.set_title('PACF')



despine([ax1,ax2])


#ARIMA

model = ARIMA(ts, order=(0, 0, 1))  

arimares = model.fit()

arimares.plot_predict(str(TS.index[262]), end=endd, alpha=0.10)

plt.plot(tsdiff)

plt.legend(loc='upper left');

print(arimares.aic, arimares.bic)