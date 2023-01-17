import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels

from statsmodels.tsa.stattools import adfuller

import pandas as pd

from fbprophet import Prophet

import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import datetime

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Basic packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd # generating random numbers

import datetime # manipulating date formats

# Viz

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots

sns.set(rc={'figure.figsize':(11, 4)})



# TIME SERIES

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs





# settings

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-iot-dataset/train_aWnotuB.csv')

test = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-iot-dataset/test_BdBKkAj_L87Nc3S.csv')

sample = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-iot-dataset/sample_submission_KVKNmI7.csv')
train.columns
train = train[['ID','DateTime','Junction','Vehicles']]

test = test[['ID','DateTime','Junction']]
train.isna().sum()
print(train.dtypes)

print(test.dtypes)
train['DateTime'] = pd.to_datetime(train['DateTime'])

test['DateTime'] = pd.to_datetime(test['DateTime'])
for junc in list(set(train['Junction'])):

    temp = train[train['Junction'] == junc]

    print(junc, temp.shape)
for junc in list(set(test['Junction'])):

    temp = test[test['Junction'] == junc]

    print(junc, temp.shape)
for junc in list(set(train['Junction'])):

    temp = train[train['Junction'] == junc]

    print("Junction", junc)

    print("Starting Date -> ",temp['DateTime'].iloc[0])

    print("Ending Date -> ", temp['DateTime'].iloc[-1])

    print()
for junc in list(set(test['Junction'])):

    temp = test[test['Junction'] == junc]

    print("Junction", junc)

    print("Starting Date -> ",temp['DateTime'].iloc[0])

    print("Ending Date -> ", temp['DateTime'].iloc[-1])

    print()
for junc in list(set(train['Junction'])):

    temp = train[train['Junction'] == junc]

    plt.figure(figsize=(30,8))

    plt.plot(temp['DateTime'],temp['Vehicles'])

    plt.show()
class StationarityTests:

    def __init__(self, significance=.05):

        self.SignificanceLevel = significance

        self.pValue = None

        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults = True):

        #Dickey-Fuller test:

        adfTest = adfuller(timeseries, autolag='AIC')

        

        self.pValue = adfTest[1]

        

        if (self.pValue<self.SignificanceLevel):

            self.isStationary = True

        else:

            self.isStationary = False

        

        if printResults:

            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])

            #Add Critical Values

            for key,value in adfTest[4].items():

                dfResults['Critical Value (%s)'%key] = value

            print('Augmented Dickey-Fuller Test Results:')

            print(dfResults)
for junc in list(set(train['Junction'])):

    temp = train[train['Junction'] == 1].copy()

    temp.rename(columns = {'DateTime':'ds','Vehicles':'y'},inplace = True)

    sTest = StationarityTests()

    sTest.ADF_Stationarity_Test(temp['y'], printResults = True)

    print("Is the time series stationary? {0}".format(sTest.isStationary))

def getResult_AdFuller_OR_kpss(_label_col ,_df, testType=1):

#     print("""

#     for dickeyFuller -> testType = 0

#     for kpss -> testType = 1

#     for acf -> testType = 2

#     for pacf -> testType = 3

#     for visual and MA ->testType = 4

#     """)

    

    if testType == 1:

        from statsmodels.tsa.stattools import adfuller

        addfull=adfuller(_df[_label_col], autolag='AIC')

        print("\n\n > Is the data stationary via addfuller test?")

        print("Test statistic = {:.3f}".format(addfull[0]))

        print("P-value = {:.3f}".format(addfull[1]))

        print("#Lag Used: = {:.3f}".format(addfull[2]))

        print("Critical values :")

        for k, v in addfull[4].items():

            print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<addfull[0] else "", 100-int(k[:-1])))



        def isStationary(tstats):

            if addfull[0] < 0.5:

                return 'TS data is stationary'

            else:

                return 'TS data is non-stationary'    

        print(isStationary(addfull[0]))

    if testType == 0:

        from statsmodels.tsa.stattools import kpss

        print("\n\n > Is the data stationary via kpss test?")

        kpss_result=kpss(_df[_label_col],regression='c')

        print("Test statistic = {:.3f}".format(kpss_result[0]))

        print("P-value = {:.3f}".format(kpss_result[1]))

        print("#Lag Used: = {:.3f}".format(kpss_result[2]))

        print("Critical values :")

        for k, v in kpss_result[3].items():

            print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<kpss_result[0] else "", 100.0-float(k[:-1])))





        def isStationary(tstats):

            if kpss_result[0] < 0.5:

                return 'TS data is stationary'

            else:

                return 'TS data is non-stationary'    

        print(isStationary(kpss_result[0]))

    if testType == 2:

        from statsmodels.graphics.tsaplots import plot_acf

        plt.figure(figsize=(20,6))

        ax= plt.subplot(111)

        plot_acf(_df[_label_col],ax=ax)

        plt.xticks(fontsize=20)

        plt.title("AutoCorrelation plot",fontsize=30,color='grey')

        plt.yticks(fontsize=20)

        plt.xlabel("#No of lags",fontsize=20)

        plt.ylabel("correlation value -1<>1",fontsize=20)

    if testType == 3:

        from statsmodels.graphics.tsaplots import plot_pacf

        plt.figure(figsize=(20,6))

        ax= plt.subplot(111)

        plot_pacf(_df[_label_col],ax=ax)

        plt.xticks(fontsize=20)

        plt.title("Partial AutoCorrelation plot",fontsize=30,color='grey')

        plt.yticks(fontsize=20)

        plt.xlabel("#No of lags",fontsize=20)

        plt.ylabel("correlation value -1<>1",fontsize=20)

        

    if testType == 4:

        print("\n\n1. use ploting to test stationarity in dataset(moving Average)")

        plt.rc('xtick', labelsize=25)     

        plt.rc('ytick', labelsize=25)

        plt.figure(figsize=(26,10))

        plt.rc('legend',fontsize=20) # using a size in points



        plt.suptitle("Rolling average(Original hourly data) to test stationarity in data", y=1.0, fontsize=30)



        # 1. Original TS Junction 1

        plt.plot(_df[_label_col],label='Orig Train Count',color='grey')



        # 2. Original TS Junction 1 Rolling mean and std

        plt.plot(_df[_label_col].rolling(window=24).mean(),label='Orig Rolling mean',color='brown' )

        plt.plot(_df[_label_col].rolling(window=24).std(),label='Orig Rolling std',color='blue' )

        plt.legend(loc='best')

        

        

def ARIMAcorrPlot(_label_col,_df):

    from statsmodels.tsa.stattools import acf, pacf 

    lag_acf = acf(_df.dropna()[_label_col], nlags=30) 

    lag_pacf = pacf(_df.dropna()[_label_col], nlags=30, method='ols')

    lag_acf,lag_pacf



    # Lets plot Autocorrelation Function

    figure = plt.figure(figsize=(25,7))

    plt.rc('xtick', labelsize=25)     

    plt.rc('ytick', labelsize=25)

    plt.rc('legend',fontsize=20) # using a size in points

    plt.plot(lag_acf) 

    plt.axhline(y=0,linestyle='--',color='gray') 

    plt.axhline(y=-1.96/np.sqrt(len(_df.dropna())),linestyle='--',color='Red',label='Lower Confidence Interval') 

    plt.axhline(y=1.96/np.sqrt(len(_df.dropna())),linestyle='--',color='Blue',label='Upper Confidence Interval') 

    plt.title('Autocorrelation Function (Give Q value on first cut point Upper CI)',fontsize=35) 

    plt.legend(loc='best')





    # Lets plot Partial Autocorrelation Function

    figure = plt.figure(figsize=(25,7))

    plt.rc('xtick', labelsize=25)     

    plt.rc('ytick', labelsize=25)

    plt.rc('legend',fontsize=20) # using a size in points

    plt.plot(lag_pacf) 

    plt.axhline(y=0,linestyle='--',color='gray') 

    plt.axhline(y=-1.96/np.sqrt(len(_df.dropna())),linestyle='--',color='red',label='Lower Confidence Interval') 

    plt.axhline(y=1.96/np.sqrt(len(_df.dropna())),linestyle='--',color='blue',label='Upper Confidence Interval') 

    plt.title('Partial Autocorrelation Function (Give P value on first cut point Upper CI)',fontsize=35) 

    plt.legend(loc='best')
train.index = train.DateTime

test.index = test.DateTime

label_col='Vehicles'
train
def applier(row):

    if row == 5 or row == 6:

        return 1

    else:

        return 0



train['year'] =train.DateTime.dt.year

train['day'] = train.DateTime.dt.day

train['month'] = train.DateTime.dt.month

train['Hour'] = train.DateTime.dt.hour

train['day of week'] = train['DateTime'].dt.dayofweek

train['weekend'] = train['DateTime'].dt.dayofweek.apply(applier)

test['year'] =test.DateTime.dt.year

test['day'] = test.DateTime.dt.day

test['month'] = test.DateTime.dt.month

test['Hour'] = test.DateTime.dt.hour

test['day of week'] = test['DateTime'].dt.dayofweek

test['weekend'] = test['DateTime'].dt.dayofweek.apply(applier)
getResult_AdFuller_OR_kpss(label_col,train,1)
getResult_AdFuller_OR_kpss(label_col,train,2)
getResult_AdFuller_OR_kpss(label_col,train,3)
getResult_AdFuller_OR_kpss(label_col,train,4)