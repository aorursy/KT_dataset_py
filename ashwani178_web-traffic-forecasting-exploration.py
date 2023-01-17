# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_1.csv').fillna(0)

train_df.head()
train_df.describe()
train_df.tail()
def get_language(page):

    res = re.search('[a-z][a-z].wikipedia.org',page)

    if res:

        return res.group()[:2]     # result fo the match converted to a str obj

    return 'na'
train_df['language'] = train_df['Page'].map(get_language)
train_df.head()
import seaborn as sb

sb.countplot(train_df['language'])
lang_sets = {}

lang_sets['en'] = train_df[train_df.language=='en'].iloc[:,0:-1]

lang_sets['ja'] = train_df[train_df.language=='ja'].iloc[:,0:-1]

lang_sets['de'] = train_df[train_df.language=='de'].iloc[:,0:-1]

lang_sets['na'] = train_df[train_df.language=='na'].iloc[:,0:-1]

lang_sets['fr'] = train_df[train_df.language=='fr'].iloc[:,0:-1]

lang_sets['zh'] = train_df[train_df.language=='zh'].iloc[:,0:-1]

lang_sets['ru'] = train_df[train_df.language=='ru'].iloc[:,0:-1]

lang_sets['es'] = train_df[train_df.language=='es'].iloc[:,0:-1]



sums = {}

for key in lang_sets:

    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
days = [ r for r in range(sums['en'].shape[0])]

def plot_with_fft(key):

    labels={'en':'English','ja':'Japanese','de':'German',

        'na':'Media','fr':'French','zh':'Chinese',

        'ru':'Russian','es':'Spanish'

       }

    fig = plt.figure(1,figsize=[15,10])

    plt.ylabel('Views per Page')

    plt.xlabel('Day')

    plt.title(labels[key])

    plt.plot(days,sums[key],label = labels[key] )

    plt.xlim(0,600)

    plt.ylim(0,8500)

    plt.show()

    

for key in sums:

    plot_with_fft(key)
plt.plot(days,sums['en'])
from statsmodels.tsa.stattools import adfuller

def test_stationarity(days,timeseries):

    

    #Determing rolling statistics

    rolmean = pd.rolling_mean(timeseries, window=12)

    rolstd = pd.rolling_std(timeseries, window=12)



    #Plot rolling statistics:

    orig = plt.plot(days,timeseries, color='blue',label='Original')

    mean = plt.plot(days,rolmean, color='red', label='Rolling Mean')

    std = plt.plot(days,rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(days,sums['en'])

test_stationarity(days,sums['zh'])
test_stationarity(days,sums['ja'])
test_stationarity(days,sums['de'])
test_stationarity(days,sums['na'])
test_stationarity(days,sums['fr'])
test_stationarity(days,sums['ru'])
test_stationarity(days,sums['es'])
from statsmodels.tsa.stattools import adfuller

def test_stationaritt(days,timeseries):

    

    #Determing rolling statistics

    rolmean = pd.rolling_mean(timeseries, window=12)

    rolstd = pd.rolling_std(timeseries, window=12)



    #Plot rolling statistics:

    orig = plt.plot(days,timeseries, color='blue',label='Original')

    mean = plt.plot(days,rolmean, color='red', label='Rolling Mean')

    std = plt.plot(days,rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)
def test_stationarit(days,timeseries): 

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
new_en=np.log(sums['en'])

new_avg_en = pd.rolling_mean(new_en,12)

plt.plot(days,new_en)

plt.plot(days,new_avg_en)
new_diff_en= new_en - new_avg_en

test_stationaritt(days,new_diff_en)
new_zh=np.log(sums['zh'])

new_avg_zh = pd.rolling_mean(new_zh,12)

plt.plot(days,new_zh)

plt.plot(days,new_avg_zh)
new_diff_zh= new_zh - new_avg_zh

test_stationaritt(days,new_diff_zh)
new_diff_zh= new_zh - new_avg_zh

new_diff_zh.dropna(inplace=True)

test_stationarit(days,new_diff_zh)
new_ja=np.log(sums['ja'])

new_avg_ja = pd.rolling_mean(new_ja,12)

plt.plot(days,new_ja)

plt.plot(days,new_avg_ja)
new_diff_ja= new_ja - new_avg_ja

test_stationaritt(days,new_diff_ja)
new_diff_ja= new_ja - new_avg_ja

new_diff_ja.dropna(inplace=True)

test_stationarit(days,new_diff_ja)
new_de=np.log(sums['de'])

new_avg_de = pd.rolling_mean(new_de,12)

plt.plot(days,new_de)

plt.plot(days,new_avg_de)
new_diff_de= new_de - new_avg_de

test_stationaritt(days,new_diff_de)
new_diff_de= new_de - new_avg_de

new_diff_de.dropna(inplace=True)

test_stationarit(days,new_diff_de)
new_fr=np.log(sums['fr'])

new_avg_fr = pd.rolling_mean(new_fr,12)

plt.plot(days,new_fr)

plt.plot(days,new_avg_fr)
new_diff_fr= new_fr - new_avg_fr

test_stationaritt(days,new_diff_fr)
new_diff_fr= new_fr - new_avg_fr

new_diff_fr.dropna(inplace=True)

test_stationarit(days,new_diff_fr)
from statsmodels.tsa.stattools import acf, pacf
#lag_acf_en = acf(new_diff_en, nlags=20)

#lag_pacf_en = pacf(new_diff_en, nlags=20, method='ols')
lag_acf_zh = acf(new_diff_zh, nlags=20)

lag_pacf_zh = pacf(new_diff_zh, nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_zh)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_zh)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_zh)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_zh)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_zh)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_zh)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
lag_acf_ja = acf(new_diff_ja, nlags=20)

lag_pacf_ja = pacf(new_diff_ja, nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_ja)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_ja)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_ja)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_ja)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_ja)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_ja)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
lag_acf_de = acf(new_diff_de, nlags=20)

lag_pacf_de = pacf(new_diff_de, nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_de)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_de)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_de)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_de)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_de)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_de)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
lag_acf_na = acf(sums['na'], nlags=20)

lag_pacf_na = pacf(sums['na'], nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_na)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(sums['na'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(sums['na'])),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_na)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(sums['na'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(sums['na'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
lag_acf_fr = acf(new_diff_fr, nlags=20)

lag_pacf_fr = pacf(new_diff_fr, nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_fr)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_fr)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_fr)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_fr)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(new_diff_fr)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(new_diff_fr)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
lag_acf_ru = acf(sums['ru'], nlags=20)

lag_pacf_ru = pacf(sums['ru'], nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_ru)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(sums['ru'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(sums['ru'])),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_ru)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(sums['ru'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(sums['ru'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
lag_acf_es = acf(sums['es'], nlags=20)

lag_pacf_es = pacf(sums['es'], nlags=20, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf_es)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(sums['es'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(sums['es'])),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf_es)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(sums['es'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(sums['es'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(new_en, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(days,new_diff_en)

plt.plot(days[1:],results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-new_diff_en)**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print (predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(new_en.ix[0], index=new_en.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(days,sums['en'])

plt.plot(days,predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sums((predictions_ARIMA-suns['en'])**2)/len(sums['en'])))