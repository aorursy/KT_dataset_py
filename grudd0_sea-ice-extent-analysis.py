%matplotlib inline
import pandas as pd       
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate,signal
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.size']=12
# function to interpolate time history data
def interp(x,y,xi):
    f = interpolate.interp1d(x, y,fill_value="extrapolate",kind='quadratic')
    # use interpolation function returned by interp1d
    return f(xi)  

# function to plot time history data
def plot_th(x,y,xs,xe,ys,ye,x_lab,y_lab):
    plt.figure(figsize=(12,6))
    plt.plot(x,y,'b')
    plt.xlim([xe,xs])
    plt.ylim([ys,ye])
    plt.grid(all)
    ax = plt.gca()
    ax.invert_xaxis()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.minorticks_on()
    plt.xlabel(x_lab);
    plt.ylabel(y_lab);
    
# function to calculate rms level time history using numpy.convolve
def window_rms(a,window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2,window,'same'))
df = pd.read_csv('../input/seaice.csv')
# get northern hemisphere data
df_north = df[df.hemisphere =='north']
# drop Missing and Source Data columns
df_north = df_north.drop(['Missing','Source Data'],axis=1)
# convert year, month,and day columns to datetime format
df_north['Date'] = pd.to_datetime(df_north[['Year','Month','Day']])
# find present time to use as reference to compute day before present
present = df_north.Date[len(df_north.index)-1]
# compute days before present and add column to dataframe
df_north['DBP'] = present - df_north.Date  
# drop year, month, and day columns from dataframe
df_north = df_north.drop(['Year','Month','Day'],axis=1)
# create numpy vectors for interpolation
day_meas = np.array(df_north.DBP.dt.days)
extent_meas = np.array(df_north.Extent)
# define analysis parameters northern hemisphere
t_start = day_meas[0]   # start interpolation "days before present"
t_end = 0 # end interpolation "days before present"
dt = 1   # interpolation delta time in days
fs = 1/dt   # samples per day
nf = 0.5 * fs  # nyquest frequency cycles per day
xlim_low = t_start # time history plot lower limit in days
xlim_high = t_end # time history plot higher limit in days
# interpolation points
time_interp = np.arange(t_start,t_end-1,-dt) 
# interpolate Sea Ice extent data
extent_interp = interp(day_meas,extent_meas,time_interp)
# plot interpolated extent data
y_lab = 'Extent 10^6 sq km'
x_lab = 'Days Before Present'
plot_th(time_interp,extent_interp,xlim_low,xlim_high,3,17,x_lab,y_lab);
plt.title('Northern Hemisphere Sea Ice Extent');
# PSD analysis for total record lenght
[freq, psd] = signal.welch(extent_interp,fs=1/dt,nperseg=4096,
                             return_onesided=True,scaling='spectrum')
# scale spectra in decibels relatice to maximum level
psd_max = max(psd)
db = 10*np.log10(psd/psd_max)
# Plot power spectral densitys
plt.figure(figsize=(12,6))
plt.semilogx(365*freq,db,'b-o')  # scale frequency in cycles per year
plt.grid(b=True, which='major', axis='both')
plt.grid(b=True, which='both', axis='x')
plt.xlabel("Frequency(cycles/year)")
plt.ylabel("Amplitude Spectrum,dB")
plt.xlim([0.1,10])
plt.ylim([-50,10]);
plt.title('Northern Hemisphere Sea Ice Extent Spectra');
extent_rms = window_rms(extent_interp,365*3)
y_lab = 'North Extent RMS, dB relation to maximum rms value'
x_lab = 'Years Before Present'
title = 'RMS Extent over running 3 year windows'
xlim_low = 37
xlim_high = 2
plt.ymin = -2.0
rms_max = max(extent_rms) # get maximum level to use as decibel reference
plot_th(time_interp/365,20*np.log10(extent_rms/rms_max),xlim_low,xlim_high,-2,0.5,x_lab,y_lab);
plt.title('Northern Hemisphere Sea Ice Extent rms time history');
from scipy.signal import butter, lfilter
from scipy.signal import freqz
# function to define a lowpass butterworth filter object
def butter_lowpass(lowcut,  fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype='lowpass')
    return b, a
# function to calculate a lowpass butterworth filter operation
def butter_lowpass_filter(data, lowcut,  fs, order):
    b, a = butter_lowpass(lowcut,  fs, order=order)
    y = lfilter(b, a, data)
    return y
# Sample rate and desired cutoff frequencies (in cycles/day).
lowcut = 0.001 # cycles/day
dt = 1  # delta time in days
fs = 1/dt  # sample rate in samples per day
# calculate lowpass filtered signal
y = butter_lowpass_filter(extent_interp, lowcut, fs, 6)
# lowpass filtered signal
plt.figure(figsize=(12,6))
plt.plot(time_interp/365, y)
plt.xlabel('time (years)')
plt.grid(True)
plt.axis('tight')
plt.xlim([37,2])
plt.ylim([8,16])
plt.title('Northern Hemisphere Sea Ice Extent low passs filtered')
plt.show()
import fbprophet
dfp = pd.DataFrame({'ds':df_north.Date,'y':df_north.Extent})
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
df_prophet.fit(dfp);
# Make a future dataframe for 2 years
df_forecast = df_prophet.make_future_dataframe(periods=365 * 2, freq='D')
# Make predictions
df_forecast = df_prophet.predict(df_forecast)
# plot forcast result
df_prophet.plot(df_forecast, xlabel = 'Date', ylabel = 'Extent');
plt.title('Northern Hemisphere Sea Ice Extent fbprophet 2 year forcast');
# Plot the trends
df_prophet.plot_components(df_forecast);
plt.figure(figsize=(12,6))
plt.plot(dfp.ds,dfp.y,'b')
plt.plot(df_forecast['ds'],df_forecast['yhat'],'r')
#plt.plot(df_forecast['ds'],df_forecast['yhat_lower'],'r--')
#plt.plot(df_forecast['ds'],df_forecast['yhat_upper'],'r-')
ts = pd.to_datetime('20000101', format='%Y%m%d', errors='coerce')
te = pd.to_datetime('20200101', format='%Y%m%d', errors='coerce')
plt.xlim([ts,te])
plt.grid(all)
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
plt.minorticks_on()
plt.xlabel('Years Before Present')
plt.ylabel('Sea Ice Extent, 10^6 sq km');
plt.legend(['measured','forcast']);
