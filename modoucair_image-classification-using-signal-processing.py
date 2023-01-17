import os

print(os.listdir("../input/chest-xray-pneumonia/chest_xray/train"))
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy import signal

from scipy.fftpack import fft, ifft

import cv2

import matplotlib.pyplot as plt

import os

from scipy import fft

from tqdm import tqdm

import seaborn as sns

from itertools import chain 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler



sns.set(style="whitegrid")

nimdata = cv2.imread("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0531-0001-0001.jpeg",cv2.IMREAD_GRAYSCALE)

pmdata = cv2.imread("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person619_virus_1190.jpeg",cv2.IMREAD_GRAYSCALE)

fig , axs = plt.subplots(1,2,figsize=(10,10))

axs[0].imshow(pmdata,cmap='gray')

axs[0].set_title('PNEUMONIA')

axs[1].imshow(nimdata,cmap='gray')

axs[1].set_title('NORMAL')

plt.show()
resize_pimg = cv2.resize(pmdata,(50,50))

resize_nimg = cv2.resize(nimdata,(50,50))

pflatten_list = list(chain.from_iterable(resize_pimg)) 

nflatten_list = list(chain.from_iterable(resize_nimg)) 

fig,axs = plt.subplots(2,1,figsize=(12,4),sharex=True)

axs[0].plot(pflatten_list)

axs[0].set_title('PNEUMONIA')

axs[1].plot(nflatten_list)

axs[1].set_title('NORMAL')

plt.show()
fig,axs = plt.subplots(1,2,figsize=(12,4),sharex=True)

sns.distplot(pflatten_list,ax=axs[0],color='Red').set_title('PNEUMONIA')

sns.distplot(nflatten_list,ax=axs[1],color='Green').set_title('NORMAL')

plt.show()
pfreqs, ppsd = signal.welch(pflatten_list)

nfreqs, npsd = signal.welch(nflatten_list)

fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)

axs[0].semilogx(pfreqs, ppsd,color ='r')

axs[1].semilogx(nfreqs, npsd,color ='g')

axs[0].set_title('PSD: PNEUMONIA')

axs[1].set_title('PSD: NORMAL')

axs[0].set_xlabel('Frequency')

axs[1].set_xlabel('Frequency')

axs[0].set_ylabel('Power')

axs[1].set_ylabel('Power')

plt.tight_layout()
sos = signal.iirfilter(3, Wn=0.01, rs=0.5 ,fs=100,btype='lp',output='sos',

                       analog=False, ftype='cheby2')

w, h = signal.sosfreqz(sos, worN=100)

plt.subplot(2, 1, 1)

db = 20*np.log10(np.maximum(np.abs(h), 1e-5))

plt.plot(w, db)

plt.ylim(-75, 5)

plt.grid(True)

plt.yticks([0, -20, -40, -60])

plt.ylabel('Gain [dB]')

plt.title('Frequency Response')

plt.subplot(2, 1, 2)

plt.plot(w/np.pi, np.angle(h))

plt.grid(True)

plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],

           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

plt.ylabel('Phase [rad]')

plt.xlabel('Normalized frequency (1.0 = Nyquist)')

plt.show()
t, s = signal.step(sos)

fig,axs = plt.subplots(1,1,figsize=(7,3),sharey=True)

axs.semilogx(t, s,color ='g')

axs.set_title('PSD')

axs.set_xlabel('Frequency')

axs.set_ylabel('Power')

plt.tight_layout()
fig, axs = plt.subplots(2, 2,figsize=(12,4), sharey=True,sharex=True)

pfiltered = signal.sosfilt(sos, pflatten_list)

nfiltered = signal.sosfilt(sos, nflatten_list)

axs[0,0].plot(pflatten_list)

axs[0,0].set_title('PNEUMONIA')

axs[0,1].plot(pfiltered)

axs[0,1].set_title('PNEUMONIA After 0.01 Hz low-pass filter')

axs[1,0].plot(nflatten_list)

axs[1,0].set_title('NORMAL')

axs[1,1].plot(nfiltered)

axs[1,1].set_title('NORMAL After 0.01 Hz low-pass filter')

#ax2.axis([0, 1, -2, 2])

plt.show()
pfreqs, ppsd = signal.welch(pfiltered)

nfreqs, npsd = signal.welch(nfiltered)

fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)

axs[0].semilogx(pfreqs, ppsd,color ='r')

axs[1].semilogx(nfreqs, npsd,color ='g')

axs[0].set_title('PSD: PNEUMONIA FILTERED')

axs[1].set_title('PSD: NORMAL FILTERED')

axs[0].set_xlabel('Frequency')

axs[1].set_xlabel('Frequency')

axs[0].set_ylabel('Power')

axs[1].set_ylabel('Power')

plt.tight_layout()
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)

pf, Pxx_den = signal.periodogram(pfiltered, 50)

f, nxx_den = signal.periodogram(nfiltered, 50)

axs[0].semilogy(pf, Pxx_den)

axs[1].semilogy(f, nxx_den)

axs[0].set_title('PNEUMOINA')

axs[1].set_title('NORMAL')

axs[0].set_xlabel('frequency [Hz]')

axs[0].set_xlabel('frequency [Hz]')

axs[0].set_ylabel('PSD [V**2/Hz]')

plt.show()
def autocorrelation(x):

    xp = np.fft.fftshift((x - np.average(x))/np.std(x))

    n, = xp.shape

    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]

    f = fft(xp)

    p = np.absolute(f)**2

    pi = ifft(p)

    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)



def kLargest(arr, k): 

    max_ks = []

    # Sort the given array arr in reverse  

    # order. 

    arr = np.sort(arr)[::-1]

    #arr.sort(reverse = True) 

    # Print the first kth largest elements 

    for i in range(k):

        max_ks.append(arr[i])

        

    return max_ks



def rms(x):

    return np.sqrt(np.mean(x**2))
print(f'Tree max values of autocorrelation\nNORMAL FILTERED:   {kLargest(autocorrelation(nfiltered), 3)}\n\nPNEUMONIA FILTERED:{kLargest(autocorrelation(pfiltered), 3)}')
ppeaks, _  = signal.find_peaks(ppsd)

npeaks, _  = signal.find_peaks(npsd)

print(f'Ten max peaksof filtered PSD\nPNEUMONIA:   {kLargest(ppeaks,10)}\nNORMAL peaks:{kLargest(npeaks,10)}')
print(f'Root Mean Square\n--------------------\nNORMAL : {rms(nfiltered)}\nPNEUMOINA : {rms(pfiltered)}')

print('--'*10)

print(f'Mean \n--------------------\nNORMAL : {np.mean(pflatten_list)}\nPNEUMOINA : {np.mean(nfiltered)}')
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)

nfft = fft(nflatten_list)

pfft = fft(pflatten_list)

axs[0].plot(pfft.real, 'b-')

axs[0].plot(pfft.imag, 'r--')

axs[0].set_title('PNEUMOINA')

axs[1].plot(nfft.real, 'b-')

axs[1].plot(nfft.imag, 'r--')

axs[1].set_title('NORMAL')

plt.legend(('real', 'imaginary'))



plt.show()
dft = cv2.dft(np.float32(nimdata),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.figure(figsize=(11,6))

plt.subplot(121),plt.imshow(nimdata, cmap = 'gray')

plt.title('ORIGINAL'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
dft = cv2.dft(np.float32(pmdata),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.figure(figsize=(11,6))

plt.subplot(121),plt.imshow(pmdata, cmap = 'gray')

plt.title('ORIGINAL'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
mpath = "../input/chest-xray-pneumonia/chest_xray/train/"

IM_SIZE = 90

from tqdm import tqdm

def get_img(folder):

    X = []

    y = []

    for xr in os.listdir(folder):

        if not xr.startswith('.'):

            if xr in ['NORMAL']:

                label = 0

            elif xr in ['PNEUMONIA']:

                label = 1

            for filename in tqdm(os.listdir(folder + xr)):

                im_array = cv2.imread(folder + xr +'/'+ filename,cv2.IMREAD_GRAYSCALE)

                if im_array is not None:

                    img = cv2.resize(im_array,(IM_SIZE,IM_SIZE))

                    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

                    dft_shift = np.fft.fftshift(dft)

                    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

                    msd = np.std(magnitude_spectrum)

                    new_image = cv2.Laplacian(img,cv2.CV_64F)

                    lpvar = abs(np.max(new_image) - np.min(new_image))/np.max(new_image)

                    #flatten the image

                    flatten_list = list(chain.from_iterable(img))

                    #filtering

                    sos = signal.iirfilter(3, Wn=0.01, rs=0.5 ,fs=100,btype='lp',output='sos',

                       analog=False, ftype='cheby2')

                    filtered = signal.sosfilt(sos, flatten_list)

                    #power Spectral density

                    _, psd = signal.welch(filtered)

                    #find peaks of PSD

                    peaks, _  = signal.find_peaks(psd)

                    maxPeaks  = kLargest(peaks, k=6)

                    #mean and rms

                    Mean = np.mean(flatten_list)

                    Rms = rms(filtered)

                    # autocorrelation

                    auto= autocorrelation(filtered)

                    maxauto = kLargest(auto, k=5)

                    #fft

                    invfft = fft(filtered)

                    vfl = np.std(flatten_list)

                    invfft_r_peaks, _  = signal.find_peaks(invfft.real)

                    invfft_imag_peaks, _  = signal.find_peaks(invfft.imag)

                    maxinvfft_r_peaks  = kLargest(invfft_r_peaks, k=6)

                    maxinvfft_imag_peaks  = kLargest(invfft_imag_peaks, k=6)

                    #peaks of periodogram filtered

                    _, Pxx_den = signal.periodogram(filtered,100)

                    Perio_Peaks, _  = signal.find_peaks(Pxx_den)

                    

                    maxPerio_Peaks  = kLargest(Perio_Peaks, k=6)

                    total = maxPeaks + [Rms,Mean,lpvar,msd,vfl] 

                    total = total + maxPerio_Peaks

                    total = total + maxinvfft_r_peaks

                    total = total + maxinvfft_imag_peaks

                    total = total + maxPeaks

                    total = total + maxauto

                    X.append(total)

                    y.append(label)    

    y = np.array(y)

    return np.array(X),y

X,y = get_img(mpath)



X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2)

#Normalization

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)    

X_test = scaler.transform(X_test)

mmscaler = MinMaxScaler()

X_train_01 = mmscaler.fit_transform(X_train)    

X_test_01 = mmscaler.transform(X_test)
import tensorflow as tf  

model = tf.keras.models.Sequential()  

model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  

model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))



model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax)) 



model.compile(optimizer='adam',  

              loss='sparse_categorical_crossentropy',  

              metrics=['accuracy'])  

model.fit(X_train_01, y_train, epochs=200)  



val_loss, val_acc = model.evaluate(X_test_01, y_test) 

print(f'Validation loss: {val_loss}')  

print(f'Validation accuracy: {val_acc}')  