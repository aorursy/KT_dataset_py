%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy import interpolate,signal
# define some analysis parameters

t_start = 2000   # start years before present for interpolation

t_end = 400000 # end years before present for interpolation

dt = 1   # delta time in years for interpolation

fs = 1/dt   # samples per year

nf = 0.5 * fs  # nyquest frequency cycles per year

xlim_low = 2000 # time history plot lower x limit in years

xlim_high = 400000 # time history plot higher x limit in years

age_interp = np.arange(t_start,t_end,dt) # interpolation points for co2 gas age

gas_age_interp = np.arange(t_start,t_end,dt) # interpolation points for temperature ice age
# function definations



# function to interpolate time history data

def interp(x,y,xi):

    f = interpolate.interp1d(x, y,fill_value="extrapolate",kind='quadratic')

    # use interpolation function returned by interp1d

    return f(xi)  



# function to plot time history data

def plot_th(x,y,xi,yi,ts,te,x_lab,y_lab,title):

    plt.figure(figsize=(20,10))

    plt.plot(x,y,'bo',xi, yi, 'r-')

    plt.xlim([ts,te])

    plt.grid(all)

    ax = plt.gca()

    ax.invert_xaxis()

    plt.grid(b=True, which='major', color='k', linestyle='-')

    plt.grid(b=True, which='minor', color='k', linestyle='--')

    plt.minorticks_on()

    plt.xlabel(x_lab);

    plt.ylabel(y_lab);

    plt.title(title);

    

# function to apply running mean filter to time historys

def mean_filter(a,n):

    return np.convolve(a, np.ones(n)/n,mode='same')
# read temp data from vostok deutnat.txt file

df_temp = pd.read_csv('../input/deutnat.csv')

# keep only age and temperature data

df_temp = df_temp.drop(['Depth corrected','deut'],axis=1)

# rename columns to age and temp for analysis

df_temp.columns = ['age','temp']

# write df_temp to numpy array

np_temp = np.array(df_temp)

# create vectors for analysis

age_meas = np_temp[:,0]

temp_meas = np_temp[:,1]

# interpolate temperature values

temp_interp = interp(age_meas,temp_meas,age_interp)

# apply mean filter to temperature data

# this should help reduce any aliasing error

n = 100 # number of points for calculating mean

temp_interp_m = mean_filter(temp_interp,n)

# plot interpolated temperature data with  mean filter applied

x_lab = 'Ice age (GT4)'

y_lab = 'Temperature difference wrt the mean recent time value'

title = 'Estimated Temperature, Last 400,000 years'

plot_th(age_meas,temp_meas,age_interp,temp_interp_m,xlim_low,xlim_high,x_lab,y_lab,title)
# read CO2 data from vostok co2nat.txt file

df_co2 = pd.read_csv('../input/co2nat.csv')

# rename columns to 'Gas age' and co2 for analysis

df_co2.columns = ['Gas age','co2']

# write df_co2 to numpy array

np_co2 = np.array(df_co2)

# create vectors for analysis

gas_age_meas = np_co2[:,0]

co2_meas = np_co2[:,1]

# interpolate co2 values

co2_interp = interp(gas_age_meas,co2_meas,gas_age_interp) 

# apply mean filter to co2 data

# this should help reduce any aliasing error

n = 100 # number of points for calculating mean

co2_interp_m = mean_filter(co2_interp,n)

# plot interpolated co2 data with  mean filter applied

x_lab = 'Gas age'

y_lab = 'CO2 (ppmv)'

title = 'CO2 (ppmv), Last 400,000 years'

plot_th(gas_age_meas,co2_meas,gas_age_interp,co2_interp_m,xlim_low,xlim_high,x_lab,y_lab,title)
# calculate cross spectral density for temp and co2 vectors

#x = y_co2[::-1]   # need to reverse co2 vector to get correct phase relation

#y = y_temp[::-1]  # need to reverse temp vector to get correct phase relation

f, Pxy = signal.csd(temp_interp_m[::-1], co2_interp_m[::-1], fs, nperseg=256512)

plt.figure(figsize=(20,10))

plt.semilogx(f, np.log(np.abs(Pxy)),'bo-')

plt.xlabel('frequency [cycles/year]')

plt.ylabel('Cross Spectral Density Amplitude[dB]')

plt.xlim([.000001,.005])

plt.ylim([.01,16])

plt.grid(all)

plt.title('Cross Spectral Density Amplitude, Temperature Deviation and CO2 ppmv');
# plot cross spectral density phase angle for temp and co2 vectors

plt.figure(figsize=(20,10))

angle = np.angle(Pxy)

plt.plot(f, angle)

plt.xlabel('frequency [cycles / year]')

plt.ylabel('Phase Angle Radians')

plt.xlim([.000001,.00006])

plt.ylim([-1,1])

plt.grid(all)

plt.title('Cross Spectral Density Phase Angle, Temperature Deviation and CO2 ppmv');
# PSD analysis

[freq, psd_1] = signal.welch(co2_interp_m,fs=1/dt,nperseg=256512,

                             return_onesided=True,scaling='spectrum')

[freq, psd_2] = signal.welch(temp_interp_m,fs=1/dt,nperseg=256512,

                             return_onesided=True,scaling='spectrum')

# Plot power spectral densitys

plt.figure(figsize=(20,10))

plt.loglog(freq,psd_1,'b-',freq,psd_2,'r--')

plt.grid()

plt.xlabel("Frequency(cycles/year)")

plt.ylabel("Amplitude Spectrum")

plt.xlim([.000005,.005])

plt.ylim([.0001,1000])

plt.legend(['CO2','Temp']);