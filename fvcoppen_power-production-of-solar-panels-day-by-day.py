from math import sqrt
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error #
solarpower = read_csv("../input/solarpower_cumuldaybyday2.csv",header = None,skiprows=1 ,names = ['date','cum_power'], sep=',',usecols = [0,1])
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
plt.xticks(color='aqua')
plt.yticks(color='aqua')
plt.show()
'''simple exponential smoothing go back to last N values
 y_t = a * y_t + a * (1-a)^1 * y_t-1 + a * (1-a)^2 * y_t-2 + ... + a*(1-a)^n * y_t-n  : c2018:Yogesh Chandra'''


def exponential_smoothing(panda_series, alpha_value):
    ouput=sum([alpha_value * (1 - alpha_value) ** i * x for i, x in enumerate(reversed(panda_series))])
    return ouput

def make_exp_smooth_col(my_df, alpha):
    my_df['smooth_power'] = 0.0
    for index in range(0, my_df.index[my_df.shape[0]-1], 1): 
        powers = my_df.day_power.head(index).values
        new_power = exponential_smoothing(powers,alpha)   # set alpha-value!!!!!!!!!
        my_df.at[index, 'smooth_power'] = new_power
        my_df.dropna()
    return my_df

my_df = solarpower.copy() #make a working copy
RMSE_list = []
steps = np.arange(0.5,0.1,-0.05)
for alpha in steps:
    make_exp_smooth_col(my_df, alpha)
    RMSE = sqrt(mean_squared_error(my_df.day_power, my_df.smooth_power))
    print('RMSE %.3f, alpha %.3f' %(RMSE, alpha))
    RMSE_list.append(RMSE)
    #print(my_df.smooth_power.describe())
    
plt.figure()
plt.plot(steps, RMSE_list)
plt.title('RMSE for line-up alpha in exponential smoothing')
plt.xticks(color='crimson')
plt.yticks(color='crimson')
plt.xlabel('Freq (days)',color='crimson')
plt.ylabel('amplitude |Y(freq)|',color='crimson')
plt.show()
    
# The optimum alpha is 0.2
'''simple exponential smoothing go back to last N values
 y_t = a * y_t + a * (1-a)^1 * y_t-1 + a * (1-a)^2 * y_t-2 + ... + a*(1-a)^n * y_t-n  : c2018:Yogesh Chandra'''

def exponential_smoothing(panda_series, alpha_value):
    ouput=sum([alpha_value * (1 - alpha_value) ** i * x for i, x in enumerate(reversed(panda_series))])
    return ouput
solarpower['smooth_power'] = 0.0
for index in range(0, solarpower.index[solarpower.shape[0]-1], 1): 
    powers = solarpower.day_power.head(index).values
    new_power = exponential_smoothing(powers,0.2)   # set alpha-value 0.025!!!!!!!!!
    solarpower.at[index, 'smooth_power'] = new_power
plt.figure(figsize=(15,7))
plt.plot(solarpower.day_power[:730])
plt.plot(solarpower.smooth_power[:730])
plt.xticks(color='aqua')
plt.yticks(color='aqua')
plt.grid()
plt.show()
solarpower= solarpower.dropna()
#signal autocorrellation on day_power
x = np.array(solarpower.day_power[:].values)
acf = []
for i in range(1, len(x)-180):
    acf.append(np.corrcoef(x[:-i], x[i:])[0,1])
plt.plot(acf)
plt.xticks(color='aqua')
plt.yticks(color='aqua')
plt.grid()
plt.show()
#stats autocorrellation day_power 
y = np.array(solarpower.day_power[:].values)

yunbiased = y-np.mean(y)
ynorm = np.sum(yunbiased**2)
acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
# use only second half
acor = acor[len(acor)//2:]

plt.plot(acor)
plt.show()
'''autocorrellation from https://stackoverflow.com/questions/643699/how-can-i-use-np-correlate-to-do-autocorrelation
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
    '''np.correlate, non partial'''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]


if __name__=='__main__':

    y=np.array(solarpower.day_power[:].values)
    #y=np.array(y).astype('float')

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
'''autocorrellation from https://stackoverflow.com/questions/643699/how-can-i-use-np-correlate-to-do-autocorrelation
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
    '''np.correlate, non partial'''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]


if __name__=='__main__':

    y=np.array(solarpower.smooth_power[:].values)
    #y=np.array(y).astype('float')

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
    plt.savefig('autocorrellation1_smooth.pdf')
    plt.show()
# simple fft 
Fs = 7*365.0;  # sampling rate 7
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the reference signal
y = np.sin(2*np.pi*ff*t)*5
y1 = np.array(solarpower.day_power[:7*365].values)
y2 = np.array(solarpower.smooth_power[:7*365].values)
n = len(y) # length of the signal
print(n,'n')
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n//2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
yf1 = np.fft.fft(y1)/(len(y1)) # fft computing and normalization
yf2 = np.fft.fft(y2)/(len(y2)) # fft computing and normalization
print('len(ffty)', len(Y))
Y = Y[range(n//2)]
print('len(Y)',len(Y))

plt.figure(figsize=(15,5))
m = 20

plt.plot(frq[1:m],abs(Y)[1:m], color='b') # plotting the spectrum
plt.annotate('reference signal 5days', color='b',xy=(1.5, 4))
plt.plot(frq[1:m],abs(yf1)[1:m],color='brown')
plt.annotate('daily power', color='brown',xy=(1.5, 3.5))
plt.plot(frq[1:m],abs(yf2)[1:m],color='purple')
plt.annotate('exp. smooth power', color='purple',xy=(1.5, 3))
plt.xticks(color='crimson')
plt.yticks(color='crimson')
plt.xlabel('Freq (days)',color='crimson')
plt.ylabel('amplitude |Y(freq)|',color='crimson')

plt.show()
