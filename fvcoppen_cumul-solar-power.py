from math import sqrt

#from sklearn.preprocessing import MinMaxScaler

import pandas as pd

#from sklearn.metrics import mean_squared_error

from pandas import read_csv

#from keras.layers import LSTM

import matplotlib.pyplot as plt

#from keras.models import Sequential

#from keras.layers import Dense

import numpy as np

#import datetime
solarpower = read_csv("../input/solarpower_2011_2018.csv",header = None,skiprows=1 ,names = ['date','cum_power'], sep=';',usecols = [0,1])

print(solarpower.head(2))
# make a column with the daily power (stationary)

solarpower['day_power']=0.0

for index in range(solarpower.index[solarpower.shape[0]-1], 0, -1):

    power = solarpower.cum_power[index] - solarpower.cum_power[index-1]

    solarpower.at[index, 'day_power']= power

#replace the day power of day 0 with the same power as day1

solarpower.at[0,'day_power'] = solarpower.at[1,'day_power']

print(solarpower.head(2))

print(solarpower.describe())
plt.plot(solarpower.day_power)

plt.show()
'''simple exponential smoothing go back to last N values

 y_t = a * y_t + a * (1-a)^1 * y_t-1 + a * (1-a)^2 * y_t-2 + ... + a*(1-a)^n * y_t-n  : c2018:Yogesh Chandra'''



def exponential_smoothing(panda_series, alpha_value):

    ouput=sum([alpha_value * (1 - alpha_value) ** i * x for i, x in enumerate(reversed(panda_series))])

    return ouput

solarpower['smooth_power'] = 0.0

for index in range(0, solarpower.index[solarpower.shape[0]-1], 1): 

    powers = solarpower.day_power.head(index).values

    new_power = exponential_smoothing(powers,0.025)   # set alpha-value!!!!!!!!!

    solarpower.at[index, 'smooth_power'] = new_power
plt.figure(figsize=(15,7))

plt.plot(solarpower.day_power[:730])

plt.plot(solarpower.smooth_power[:730])

plt.xticks(color='aqua')

plt.yticks(color='aqua')

plt.grid()

plt.show()
solarpower= solarpower.dropna()
x = np.array(solarpower.day_power[:].values)

acf = []

for i in range(1, len(x)-180):

    acf.append(np.corrcoef(x[:-i], x[i:])[0,1])

plt.plot(acf)

plt.xticks(color='aqua')

plt.yticks(color='aqua')

plt.grid()

plt.show()
#calc autocorrellation

y = np.array(solarpower.day_power[:].values)



yunbiased = y-np.mean(y)

ynorm = np.sum(yunbiased**2)

acor = np.correlate(yunbiased, yunbiased, "same")/ynorm

# use only second half

acor = acor[len(acor)//2:]



plt.plot(acor)

plt.show()
'''autocorrellation from https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation

see Jason code'''



def autocorr1(x,lags):

    '''np.corrcoef, partial'''



    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]

    return np.array(corr)



def autocorr2(x,lags):

    '''manualy compute, non partial'''



    mean=np.mean(x)

    var=np.var(x)

    xp=x-mean

    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]



    return np.array(corr)



def autocorr3(x,lags):

    '''fft, pad 0s, non partial'''



    n=len(x)

    # pad 0s to 2n-1

    ext_size=2*n-1

    # nearest power of 2

    fsize=2**np.ceil(np.log2(ext_size)).astype('int')



    xp=x-np.mean(x)

    var=np.var(x)



    # do fft and ifft

    cf=np.fft.fft(xp,fsize)

    sf=cf.conjugate()*cf

    corr=np.fft.ifft(sf).real

    corr=corr/var/n



    return corr[:len(lags)]



def autocorr4(x,lags):

    '''fft, don't pad 0s, non partial'''

    mean=x.mean()

    var=np.var(x)

    xp=x-mean



    cf=np.fft.fft(xp)

    sf=cf.conjugate()*cf

    corr=np.fft.ifft(sf).real/var/len(x)



    return corr[:len(lags)]



def autocorr5(x,lags):

    '''numpy.correlate, non partial'''

    mean=x.mean()

    var=np.var(x)

    xp=x-mean

    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)



    return corr[:len(lags)]





if __name__=='__main__':



    y=np.array(solarpower.day_power[:].values)

    

    lags=range(7*365)

    fig,ax=plt.subplots(figsize=(15,15))



    for funcii, labelii in zip([autocorr1, autocorr2, autocorr3, autocorr4,

        autocorr5], ['np.corrcoef, partial', 'manual, non-partial',

            'fft, pad 0s, non-partial', 'fft, no padding, non-partial',

            'np.correlate, non-partial']):



        cii=funcii(y,lags)

        #print(labelii)

        #print(cii)

        ax.plot(lags,cii,label=labelii)



    ax.set_xlabel('lag')

    ax.set_ylabel('correlation coefficient')

    ax.legend()

    plt.savefig('autocorrellation1.pdf')

    plt.show()