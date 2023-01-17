# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







import os

import pandas as pd

import math

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import statsmodels.api as sm



### CONSTANTS USED 

TRAIN_PERC=0.90

PLT_WIDTH=5

PLT_HEIGHT=9

PLT_DPI=120

###

data=[]

file=None

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        file=os.path.join(dirname,filename)



#print(file)







#load the data and generate the time series

#interested only in the third column

df=pd.read_csv(file,delim_whitespace=True,header=None,names=['J1','J2','Y','J3','J4'])

df.head()

time_series=df.iloc[:,2]

#remove Nan and non-digits

time_series.dropna(inplace=True)

time_series=time_series.drop([0,1])





time_series=time_series.astype(float)

time_series=time_series.reset_index(drop=True)

##CREATE TRAIN AND TEST DATA 

###

trainlen=math.floor(time_series.size*TRAIN_PERC)

test_series=time_series[trainlen:]

train_series=time_series[:trainlen]



###





#display the time series

#https://machinelearningmastery.com/time-series-data-visualization-with-python/

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(PLT_HEIGHT,PLT_WIDTH), 'figure.dpi':PLT_DPI})

plt.rcParams.update({'font.size':9})

plt.rc('figure', titlesize=9)

#Line Plot





fig,ax=plt.subplots(2,2)

train_series.plot(ax=ax[0,0])

train_series.hist(ax=ax[0,1])

plot_acf(train_series,ax=ax[1,0],title='Auto')

plot_pacf(train_series,ax=ax[1,1],title='')



plt.show()





def difference(series,order):

    """

       Parameters:

                    Input:

                        series = input time series

                        order= order of differencing

                        

                    Returns:

                        returns a series consisting difference values of order=order

        

      

    """

    

    if order == 0:

        return series

    else:

        diff=series.diff(-1)

        order_=order-1

        return difference(series=diff,order=order_)


zod=difference(train_series,0)   #zeroth-order differencing = original series

zod=zod.rename('0')

fod=difference(train_series,1)   #first-order

fod=fod.rename('1')

sod=difference(train_series,2)   #second-order

sod=sod.rename('2')

tod=difference(train_series,3)   #third-order

tod=tod.rename('3')

#print(type(zod))

ll=[zod,fod,sod,tod]



#keys=[s.name for s in fod]

#print(zod.name)

df=pd.concat([zod,fod,sod,tod],axis=1,keys=[s.name for s in ll])

print(df.head())
df.describe()
from pandas.plotting import lag_plot



def six_plots(sr):

    

    sr=sr.dropna()

    plt.rcParams.update({'figure.figsize':(PLT_HEIGHT,PLT_WIDTH), 'figure.dpi':PLT_DPI})

    fontdict={'fontsize':9,'verticalalignment':'bottom'}

    fig,ax=plt.subplots(2,3)

    sr.plot(ax=ax[0,0])  #plot the series

    sr.hist(ax=ax[0,1]) #must be gaussian like

    sm.qqplot(sr,ax=ax[0,2],line='45') # how close does the series fit the normal distribution

    lag_plot(sr,ax=ax[1,0]) #lag-1 plot to see autocorrelations   

    plot_acf(sr,ax=ax[1,1],title='') #acf plot

    plot_pacf(sr,ax=ax[1,2],title='') #pacf plot

    

    #set the titles in the correct place. 

    #https://matplotlib.org/gallery/pyplots/text_layout.html#sphx-glr-gallery-pyplots-text-layout-py

    left = 0.45

    bottom = -0.5

    top = 1.2

    

    #for the top 3 plots, titles are on the top

    ax[0,0].text(left, top, 'run sequence',

        horizontalalignment='left',

        verticalalignment='top',

        transform=ax[0,0].transAxes)

    ax[0,1].text(left, top, 'hist',

        horizontalalignment='left',

        verticalalignment='top',

        transform=ax[0,1].transAxes)

    ax[0,2].text(left, top, 'Q-Q',

        horizontalalignment='left',

        verticalalignment='top',

        transform=ax[0,2].transAxes)

    ax[0,2].set_xlabel('')

    ax[0,2].set_ylabel('')

    

    #for the bottom 3 plots , titles are at the bottom

    ax[1,0].text(left, bottom, 'Lag-plot',

        horizontalalignment='left',

        verticalalignment='bottom',

        transform=ax[1,0].transAxes)

    ax[1,1].text(left, bottom, 'ACF',

        horizontalalignment='left',

        verticalalignment='bottom',

        transform=ax[1,1].transAxes)    

    ax[1,2].text(left, bottom, 'PACF',

        horizontalalignment='left',

        verticalalignment='bottom',

        transform=ax[1,2].transAxes)

    

    fig.tight_layout()

    fig.suptitle('')

    plt.show()

    

import matplotlib.gridspec as gridspec



six_plots(df['0'])

six_plots(df['1'])

six_plots(df['2'])
from statsmodels.tsa.stattools import adfuller



def ADF(sr):

    """

    Augmented Dickey Fuller Test.

    """

    sr=sr.dropna() #remove any invalids

    results=adfuller(sr)

    print('ADF statistic: ',results[0])

    print('p-value: ',results[1]) #probability that the null hypothesis is true

    print('Critical vals: ')

    for i,j in results[4].items():

        print(i,j)



print('difference order = 1')        

ADF(df['1'])

print()

print('difference order = 2')      

ADF(df['2'])

#print('Aresults)
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox



from scipy.stats import chi2





def chi_square_table(p,dof):

    """

    https://stackoverflow.com/questions/32301698/how-to-build-a-chi-square-distribution-table

    

    Parameters:

            Input:

                p= p-value 

                dof = degree of freedom

            Returns:

                chi-sq critical value corresponding to (p,dof)

    

    """

    return chi2.isf(p,dof)





def chi_sq_critical_val(alpha,dof):

    """

    return the critical val (c) for chi-sq distrib parameterized by 

    probability(pr)=1-alpha and degrees of freedom=dof 

    c is the value at and below which pr% of data exists

    

    """

    pr=1-alpha

    val=chi2.ppf(pr,dof)

    return val



    





def eval_arima(series,order,lags,dynamic=False,alpha=0.05):

    """

    1.fit the model 

    2.get the residuals

    3.plot the residuals

    4.does it look like white noise? mean=0, normally distributed?

    5.calculate Q on the residuals for number of lags

    6.choose a level of significance

    7.choose degrees of freedom

    8.calculate the critical value of the chi-sq statistic

    9.accept or reject null hypothesis



    Parameters:

            Input:

                series          = time series to be fit by the ARIMA model

                order           = 3-tuple of form (p,d,q) where p=AR terms. d=order of differencing, q= MA terms of an ARIMA(p,d,q) model

                dynamic         = True ==> out-of-sample (predict unseen (test) data),

                                  False ==> in-sample  (predict on the data trained on)

                alpha           = significance level

                lags            = number of lags used to calculate the Ljung-Box Q statistic

            Return:

                    fitted model

    """





    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})



    #fit the model

    model=ARIMA(series,order=order)   #ARIMA(0,1,1) model

    model_fit=model.fit(disp=-1)



    #print(type(model_fit))

    print(model_fit.summary())

    

    #display the fit of the model

    model_fit.plot_predict(dynamic=dynamic).suptitle("model fit on training data")

    plt.show()

    

   

    #get the residuals

    residuals=model_fit.resid

    #plot the residuals

    fig,ax=plt.subplots(1,2)



    residuals.plot(title='Residuals',ax=ax[0])

    residuals.plot(kind='kde',title='probability distribution of residuals',ax=ax[1])

    #print(model_fit.)

    plt.show()

    

    #are the residuals random?

    print(residuals.describe())

    #autocorrelation plots of residuals

    six_plots(residuals)

   

    #Significance Level at 5%

    #alpha=0.05



    #The Ljung-Box Test 

    Q,p=ljungbox(residuals,range(1,lags),boxpierce=False)

    c=[]

    for i in range(len(Q)):

        dof=i+1                

        c.append(chi_sq_critical_val(alpha,dof))

        #print('Chi-statistic(Q) :',Q[i],'  p-value:',p[i],'   critical value: ',c," KEEP H0" if Q[i]<c else "DNT KEEP H0")

    

    #plot Q versus c

    #accept if Q stays below the 45 deg line i.e Q<c

    arstr="ARIMA"+str(order)+""

    plt.plot(c,Q,label=arstr)

    plt.plot(c,c,label='c=Q')

    plt.xlabel('Q values')

    plt.ylabel('critical values')

    plt.title('Ljung - Box Test')

    plt.legend()

    plt.show()

    return model_fit

    





arima_011=eval_arima(train_series,order=(0,1,1),lags=25)

arima_210=eval_arima(train_series,order=(2,1,0),lags=25)

arima_211=eval_arima(train_series,order=(2,1,1),lags=25)

def arima_forecast(model,test_sr,train_sr):

    """

    Forecast arima models on the test series (test_sr)

    Parameters:

        Input:

            model= arima model used for forecasting

            test_sr = test series for forecasting

            train_sr= training data used to build model

        Returns:

            dictionary containing metric values

            

    """

    fc,se,cf= model.forecast(test_sr.size,alpha=0.05)

    #Convert to series



    fc_series=pd.Series(fc,index=test_sr.index)

    lower_cf=pd.Series(cf[:,0],index=test_sr.index)

    upper_cf=pd.Series(cf[:,1],index=test_sr.index)



    #plotting

    plt.plot(train_sr,label='training')

    plt.plot(test_sr,label='test')

    plt.plot(fc_series,label='forecast')

    plt.fill_between(lower_cf.index,lower_cf,upper_cf,

                    color='k',alpha=0.15)

    plt.legend()

    plt.show()

    

    #forecast accuracies

    actual=test_sr.values

    forecast=fc_series

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    maxs = np.amax(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe, 'rmse':rmse,

            'corr':corr, 'minmax':minmax})

    

    

arima011_metrics=arima_forecast(arima_011,test_sr=test_series,train_sr=train_series)

arima210_metrics=arima_forecast(arima_210,test_sr=test_series,train_sr=train_series)

arima211_metrics=arima_forecast(arima_211,test_series,train_series)
lm=[arima011_metrics,arima210_metrics,arima211_metrics]

dlmdf=pd.DataFrame(lm)

dlmdf.head()
   

f,axx=plt.subplots(3,3)    

dlmdf['mape'].plot(ax=axx[0,0])

dlmdf['me'].plot(ax=axx[0,1])

dlmdf['mae'].plot(ax=axx[0,2])

dlmdf['mpe'].plot(ax=axx[1,0])

dlmdf['rmse'].plot(ax=axx[1,1])

dlmdf['corr'].plot(ax=axx[1,2])

dlmdf['minmax'].plot(ax=axx[2,0])

#axx[2,1].setvisible(False)

f.delaxes(axx[2,1])

f.delaxes(axx[2,2])

f.tight_layout()