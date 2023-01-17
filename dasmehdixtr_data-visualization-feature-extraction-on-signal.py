import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/carla-driver-behaviour-dataset/full_data_carla.csv', index_col=0)

data.info()
data.shape
time = np.arange(0,data.shape[0],1)

fig, (ax1,ax2, ax3, ax4,ax5,ax6) = plt.subplots(6, 1,figsize=(10,15))

fig.suptitle('Sensor Data Visualization',fontsize=16)

fig.tight_layout()



ax1.plot(time, data['accelX'],c='r') 

ax1.grid(alpha=0.1)

ax1.set_title("X-axis of Accelerometer")



ax2.plot(time, data['accelY'],c='g')

ax2.grid(alpha=0.1)

ax2.set_title("Y-axis of Accelerometer")



ax3.plot(time, data['accelZ'],c='b')

ax3.grid(alpha=0.1)

ax3.set_title("Z-axis of Accelerometer")



ax4.plot(time, data['gyroX'],c='r')

ax4.grid(alpha=0.1)

ax4.set_title("X-axis of Gyroscope")



ax5.plot(time, data['gyroY'],c='g')

ax5.grid(alpha=0.1)

ax5.set_title("Y-axis of Gyroscope")



ax6.plot(time, data['gyroZ'],c='b')

ax6.grid(alpha=0.1)

ax6.set_title("Z-axis of Gyroscope")
data_person_1 = data[(data['class'] == 'mehdi')]

data_person_2 = data[(data['class'] == 'hurcan')]

time_p1 = np.arange(0,data_person_1.shape[0],1)

time_p2 = np.arange(0,data_person_2.shape[0],1)
fig, ax = plt.subplots(6, 2,figsize=(8,12), constrained_layout=True)

fig.suptitle('Does signals "visually" looks seperable?')



ax[0][0].plot(time_p1,data_person_1['accelX'])

ax[0][1].plot(time_p2,data_person_2['accelX'])



ax[1][0].plot(time_p1,data_person_1['accelY'])

ax[1][1].plot(time_p2,data_person_2['accelY'])



ax[2][0].plot(time_p1,data_person_1['accelZ'])

ax[2][1].plot(time_p2,data_person_2['accelZ'])



ax[3][0].plot(time_p1,data_person_1['gyroX'])

ax[3][1].plot(time_p2,data_person_2['gyroX'])



ax[4][0].plot(time_p1,data_person_1['gyroY'])

ax[4][1].plot(time_p2,data_person_2['gyroY'])



ax[5][0].plot(time_p1,data_person_1['gyroZ'])

ax[5][1].plot(time_p2,data_person_2['gyroZ'])

data_apo = data[data['class']=='apo']

acc_apo_x = data_apo.iloc[:,0]

data_hrcn = data[data['class']=='hurcan']

acc_hrcn_x = data_hrcn.iloc[:,0]
from scipy import signal

autocorr = signal.fftconvolve(acc_apo_x,acc_apo_x[::-1],mode='full')

autocorr2 = signal.fftconvolve(acc_hrcn_x,acc_hrcn_x[::-1],mode='full')



import matplotlib.pyplot as plt

fig, (ax_orig, ax_mag) = plt.subplots(2, 1)

ax_orig.plot(acc_apo_x)

ax_orig.plot(acc_hrcn_x)

ax_orig.set_title('White noise')

ax_mag.plot(np.arange(-len(acc_apo_x)+1,len(acc_apo_x)), autocorr)

ax_mag.plot(np.arange(-len(acc_hrcn_x)+1,len(acc_hrcn_x)), autocorr2)

ax_mag.set_title('Autocorrelation')

fig.tight_layout()

fig.show()
labels = data['class'].unique()

print(labels)
import matplotlib.pyplot as plt

for label in labels:

    data_label = data[data['class']==label]

    acc_label_x = data_label.iloc[:,0]

    autocorr = signal.oaconvolve(acc_label_x,acc_label_x[::-1],mode='full') #alternatively you can use fftconvolve function

    plt.plot(np.arange(-len(acc_label_x)+1,len(acc_label_x)), autocorr)

plt.title('AutoCorrelations of drivers for AccelX')

plt.show()
import matplotlib.pyplot as plt

for label in labels:

    data_label = data[data['class']==label]

    acc_label_x = data_label.iloc[:,0]

    plt.plot(np.arange(0,len(acc_label_x)), acc_label_x,label=label)

plt.legend()

plt.title('Visualization of drivers for AccelX')

plt.show()
for label in labels:

    data_label = data[data['class']==label]

    acc_label_x = data_label.iloc[:,0]

    corr = signal.correlate(acc_label_x,np.ones(len(acc_label_x)),mode='same') / len(acc_label_x)

    clock= np.arange(64, len(acc_label_x), 128)

    plt.plot(clock, corr[clock],label= label)

plt.legend()

plt.title('cross-correlation-features for AccelX')

plt.show()
fs = 20

f, t, Sxx = signal.spectrogram(acc_apo_x, fs)

f1, t1, Sxx1 = signal.spectrogram(acc_hrcn_x,fs)

fig, (ax_1, ax_2) = plt.subplots(2, 1)

ax_1.pcolormesh(t, f, Sxx, shading='gouraud')

ax_1.set_title('Spectrograms of two drivers')

ax_2.pcolormesh(t1, f1, Sxx1, shading='gouraud')

fig.tight_layout()

fig.show()
import numpy as np

t = np.linspace(0,1,len(acc_apo_x))

b,a = signal.butter(3,0.05)

zi = signal.lfilter_zi(b,a)

z, _ = signal.lfilter(b,a, acc_apo_x, zi=zi*acc_apo_x[0])

z2,_ = signal.lfilter(b,a,z,zi=zi*z[0])

y = signal.filtfilt(b,a,acc_apo_x)



plt.figure

plt.plot(t, acc_apo_x, 'b', alpha=0.75)

plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')

plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',

            'filtfilt'), loc='best')

plt.grid(True)

plt.title('Filtering Example on driver {} AccelX'.format('apo'))

plt.show()