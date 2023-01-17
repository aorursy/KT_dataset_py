import pywt
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
df_tmsi = pd.read_csv('../input/refa-ecg/refa_anirban_24_S12.csv')
df_tmsi = df_tmsi.reset_index() 
df_tmsi.head()
df_tmsi.info()

start = 0
end = 10000
signal = df_tmsi['ML2'][start:end].values
time = df_tmsi['index'][start:end].values
time = time * (1/2048)

plt.figure(figsize=(25,10))
plt.plot(time, signal)
plt.title('Signal', size=15)
plt.show()
df_ecg = pd.read_csv('../input/maheenecg/ECG12_subject_hossain.csv')
df_ecg.head()
df_ecg.info()

start = 0
n_times = 10000
signal = df_ecg['lead2'][start:n_times].values
time = df_ecg['sample'][start:n_times].values

plt.figure(figsize=(25,10))
plt.plot(signal)
plt.title('Signal', size=15)
plt.show()
coeffs = pywt.wavedec(signal, 'db6', level=5)
cA5, cD5, cD4, cD3, cD2, cD1= coeffs

cD5 = cD5*0
cD4 = cD4*0
cD3 = cD3*0
cD2 = cD2*0
cD1 = cD1*0

recoeffs=[cA5, cD5, cD4, cD3, cD2, cD1]

reECG = pywt.waverec(recoeffs, 'db6')

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(time, signal, color = "b", alpha = 0.5, label = 'original signal')
ax.plot(time, reECG, 'k', label = 'DWT smoothing', linewidth=2)
ax.legend()
ax.set_title('Removing PowerLine Noise with DWT', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()
data= signal
waveletname = 'db6'

fig, coordinate=plt.subplots(nrows=1,ncols = 2, figsize = (25, 2))
coordinate[0].plot(data,'r')
coordinate[1].plot(data,'g')

fig, axarr = plt.subplots(nrows = 8, ncols = 2, figsize = (25,20))
for ii in range(8):
    (data, coeff_d) = pywt.dwt(reECG, waveletname)    # Change the DATA
    axarr[ii, 0].plot(data,'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii+1),fontsize = 14, rotation = 90)
    axarr[ii, 0].set_yticklabels([])
    if ii==0:
        axarr[ii,0].set_title("Approximation coefficients",fontsize = 14)
        axarr[ii,1].set_title("Detail coefficients", fontsize = 14)
    axarr[ii,1].set_yticklabels([])
plt.tight_layout()
plt.show()
coeffs = pywt.wavedec(data, 'db4', level=8)
cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

cD8=cD8*0    
cD7=cD7*0
cD6=cD6*0
#cD3=cD3*0
cD2=cD2*0
cD1=cD1*0
cA8=cA8*0

recoeffs=[cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]

rwave = pywt.waverec(recoeffs, 'db4')
rwave = np.square(rwave)

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(time, signal, color = "b", alpha = 0.5, label = 'original signal')
# ax.plot(time, rwave, 'k', label = 'R wave', linewidth=2)
# ax.legend()
# ax.set_title('Matching R Wave', fontsize=18)
# ax.set_ylabel('Signal Amplitude', fontsize=16)
# ax.set_xlabel('Sample No', fontsize=16)
# plt.show()
fig, axarr = plt.subplots(nrows= 2, ncols= 1, figsize= (25, 10))
axarr[0].plot(time, signal,'r')
axarr[1].plot(rwave,'g')
fig, axarr = plt.subplots(nrows= 2, ncols= 1, figsize= (25, 10))
axarr[0].plot(time, signal,'r')
axarr[1].plot(rwave,'g')
peaks, _ = find_peaks(rwave)
fig, axarr = plt.subplots(nrows= 2, ncols= 1, figsize= (25, 10))
axarr[0].plot(signal,'r')
axarr[1].plot(rwave,'g')
plt.plot(signal)
plt.plot(time[peaks], signal[peaks], "x")
plt.plot(np.zeros_like(signal), "--", color="gray")
plt.show()