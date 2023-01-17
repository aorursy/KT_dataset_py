import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import matplotlib.image as mpimg

import io

import librosa

import warnings

warnings.filterwarnings('ignore') # get rid of librosa warnings
import math



# Continous Wavelet Transform with Morlet wavelet 

# Original code by Alexander Neergaard, https://github.com/neergaard/CWT

# 

# Parameters:

#   data: input data

#   nv: # of voices (scales) per octave

#   sr: sampling frequency (Hz)

#   low_freq: lowest frequency (Hz) of interest (limts longest scale)

def cwt2(data, nv=10, sr=1., low_freq=0.):

    data -= np.mean(data)

    n_orig = data.size

    ds = 1 / nv

    dt = 1 / sr



    # Pad data symmetrically

    padvalue = n_orig // 2

    x = np.concatenate((np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))

    n = x.size



    # Define scales

    _, _, wavscales = getDefaultScales(n_orig, ds, sr, low_freq)

    num_scales = wavscales.size



    # Frequency vector sampling the Fourier transform of the wavelet

    omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)

    omega *= (2 * np.pi) / n

    omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))



    # Compute FFT of the (padded) time series

    f = np.fft.fft(x)



    # Loop through all the scales and compute wavelet Fourier transform

    psift, freq = waveft(omega, wavscales)



    # Inverse transform to obtain the wavelet coefficients.

    cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)

    cfs = cwtcfs[:, padvalue:padvalue + n_orig]

    freq = freq * sr



    return cfs, freq



def getDefaultScales(n, ds, sr, low_freq):

    nv = 1 / ds

    # Smallest useful scale (default 2 for Morlet)

    s0 = 2



    # Determine longest useful scale for wavelet

    max_scale = n // (np.sqrt(2) * s0)

    if max_scale <= 1:

        max_scale = n // 2

    max_scale = np.floor(nv * np.log2(max_scale)) 

    a0 = 2 ** ds

    scales = s0 * a0 ** np.arange(0, max_scale + 1)

    

    # filter out scales below low_freq

    fourier_factor = 6 / (2 * np.pi)

    frequencies = sr * fourier_factor / scales

    frequencies = frequencies[frequencies >= low_freq]

    scales = scales[0:len(frequencies)]



    return s0, ds, scales



def waveft(omega, scales):

    num_freq = omega.size

    num_scales = scales.size

    wft = np.zeros([num_scales, num_freq])



    gC = 6

    mul = 2

    for jj, scale in enumerate(scales):

        expnt = -(scale * omega - gC) ** 2 / 2 * (omega > 0)

        wft[jj, ] = mul * np.exp(expnt) * (omega > 0)



    fourier_factor = gC / (2 * np.pi)

    frequencies = fourier_factor / scales



    return wft, frequencies
import scipy.fftpack

from scipy import signal

from scipy.signal import chirp



DB_RANGE = 100 # dynamic range to show in dB

SR = 22050

CMAP = 'magma'



def show_sigx3(d):

    fig, axes = plt.subplots(1, 3, figsize=(16,5))

    # FFT

    N, T = SR, 1./SR

    x = np.linspace(0.0, int(N*T), N)

    yf = scipy.fftpack.fft(d*np.hamming(len(d)))

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    axes[0].plot(xf,  20*np.log10(2.0/N *np.abs(yf[:N//2])))

    axes[0].set_ylim(-80,0)

    axes[0].set_title('FFT')

    # Spectrogram

    f, t, Sxx = signal.spectrogram(d, SR)

    axes[1].pcolormesh(t, f, 20*np.log10(Sxx), shading='auto', cmap=CMAP, vmax=-60, vmin=-60-DB_RANGE)

    axes[1].set_title('Spectrogram')

    # CWT

    #cs, f = calc_cwt(d)

    cs, f = cwt2(d, nv=12, sr=SR, low_freq=40)

    axes[2].imshow(20*np.log10(np.abs(cs)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-DB_RANGE)

    axes[2].set_title('Scaleogram')

    plt.show()
d = np.random.randn(SR)

show_sigx3(d)
t = np.linspace(0, 1, SR)

d = chirp(t, f0=50, f1=10e3, t1=1., method='hyperbolic')

show_sigx3(d)
from IPython.display import Audio

from sklearn.preprocessing import minmax_scale



TRAIN_DIR = '../input/birdsong-recognition/train_audio'

SR = 22050 # sample rate in Hz



def rd_file(fname, offset=0, duration=60):

    data, _ = librosa.load(fname, sr=SR, mono=True, offset=offset, duration=duration)

    data = minmax_scale(data-data.mean(), feature_range=(-1,1))

    return data
# load a file with Black-throated Blue Warbler

d1 = rd_file(TRAIN_DIR+'/btbwar/XC139608.mp3')

Audio(d1, rate=SR)
# load another file with Black-throated Green Warbler

d2 = rd_file(TRAIN_DIR+'/btnwar/XC135117.mp3')

Audio(d2, rate=SR)
cs1, f1 = cwt2(d1, nv=12, sr=SR, low_freq=40)

plt.figure(figsize = (16,4))

plt.imshow(20*np.log10(np.abs(cs1)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-30)

plt.show()
cs2, f2 = cwt2(d2, nv=12, sr=SR, low_freq=40)

plt.figure(figsize = (16,4))

plt.imshow(20*np.log10(np.abs(cs2)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-30)

plt.show()
def plot_sigx2(d1, d2, name1='data 1', name2='data 2', cwt=True, db_range=30):

    fig, axes = plt.subplots(1, 2, figsize=(16,4))

    d = [d1, d2]

    name = [name1, name2]

    for i in range(2):

        if cwt == True:

            cs, _ = cwt2(d[i], nv=12, sr=SR, low_freq=40)

            axes[i].imshow(20*np.log10(np.abs(cs)),  cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-db_range)

        else:

            f, t, Sxx = signal.spectrogram(d[i], SR)

            axes[i].pcolormesh(t, f, 20*np.log10(Sxx), shading='auto', cmap=CMAP, vmax=-60, vmin=-60-db_range)

        axes[i].set_title(name[i])

    plt.show()
plot_sigx2(d1[25000:65000], d2[45000:85000], 

           name1='Black-throated Blue Warbler', name2='Black-throated Green Warbler')
plot_sigx2(d1[25000:65000], d2[45000:85000], 

           name1='Black-throated Blue Warbler', name2='Black-throated Green Warbler', cwt=False, db_range=60)
plot_sigx2(d1[25000:65000], d2[45000:85000], 

           name1='Black-throated Blue Warbler', name2='Black-throated Green Warbler', db_range=120)
plot_sigx2(d1[25000:65000], d2[45000:85000], 

           name1='Black-throated Blue Warbler', name2='Black-throated Green Warbler', cwt=False, db_range=120)
d3 = rd_file(TRAIN_DIR+'/normoc/XC54018.mp3', offset=127, duration=10)

Audio(d3, rate=SR)
cs3a, f3 = cwt2(d3, nv=12, sr=SR, low_freq=40)

cs3b=np.zeros((97,10*SR), dtype=np.complex)

for i in range(10):

    cs3b[:,i*SR:(i+1)*SR], _ = cwt2(d3[i*SR:(i+1)*SR], nv=12, sr=SR, low_freq=40)

fig, axes = plt.subplots(3, 1, figsize=(16,12))

axes[0].imshow(20*np.log10(np.abs(cs3a)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-40)

axes[0].set_title('Scaleogram - 10s')

axes[1].imshow(20*np.log10(np.abs(cs3b)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-40)

axes[1].set_title('Scaleogram - 10 x 1s')

diff = np.abs(np.abs(cs3a)-np.abs(cs3b))

diff[diff == 0.] = 1e-15

axes[2].imshow(20*np.log10(diff), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-40)

axes[2].set_title('Scaleogram - difference')

plt.show()
f, t, Sxxa = signal.spectrogram(d3, SR)

Sxxb=np.zeros((129,10*98))

for i in range(10):

    _, _, Sxxb[:,i*98:(i+1)*98] = signal.spectrogram(d3[i*SR:(i+1)*SR])

fig, axes = plt.subplots(3, 1, figsize=(16,12))

axes[0].pcolormesh(t[0:980], f, 20*np.log10(Sxxa[:,0:980]), shading='auto', cmap=CMAP, vmax=-90, vmin=-130)

axes[0].set_title('Spectrogram - 10s')

axes[1].pcolormesh(t[0:980], f, 20*np.log10(Sxxb), shading='auto', cmap=CMAP, vmax=-0, vmin=-40)

axes[1].set_title('Spectrogram - 10 x 1s')

diff = np.abs(np.abs(Sxxa[:,0:980]*3e4)-np.abs(Sxxb))

diff[diff == 0.] = 1e-15

axes[2].pcolormesh(t[0:980], f, 20*np.log10(diff), shading='auto', cmap=CMAP, vmax=-0, vmin=-40)

axes[2].set_title('Spectrogram - difference')

plt.show()
# calculate variance of coefficients

def calc_var(cs, thres):

    c = 20*np.log10(np.abs(cs))

    c[c < thres] = 0.

    e = np.var(c, axis=0)

    return e / max(e)
fig, axes = plt.subplots(2, 1, figsize=(16,4))

v1 = calc_var(cs1, -30)

axes[0].plot(v1)

v2 = calc_var(cs2, -30)

axes[1].plot(v2)

plt.show()
from scipy.signal import find_peaks



def mask_sig(n, peaks, sr=22050, dur=0.1):

    mask = np.zeros(n)

    subm = int(sr*dur*0.5)

    if len(peaks > 0):

        for i in range(len(peaks)):

            mask[max(peaks[i]-subm, 0): min(peaks[i]+subm, n)] = 1

    return mask



fig, axes = plt.subplots(2, 1, figsize=(16,4))

# peak detection + gliding window

peaks, _ = find_peaks(v1, prominence=0.3)

axes[0].plot(v1)

axes[0].plot(peaks, v1[peaks], "x")

m = mask_sig(len(v1), peaks, SR, 0.3)

axes[1].plot(m)

plt.show()
def get_mask(vdata, prom=0.2, dur=0.2, sr=22050):

    peaks, _ = find_peaks(vdata, prominence=prom)

    return mask_sig(len(vdata), peaks, sr, dur)

    

def get_regions(mask, sr, species, filename):

    regions = scipy.ndimage.find_objects(scipy.ndimage.label(mask)[0])

    regs = []

    for r in regions:

        dur = round((r[0].stop-r[0].start)/sr,3)

        regs.append([r[0].start, r[0].stop, dur,species,filename])

    return pd.DataFrame(regs, columns=['Start', 'End', 'Duration','Species','File'])
mask = get_mask(v1, prom=0.3, dur=0.2, sr=SR)

df = get_regions(mask, SR, 'btbwar', 'XC139608.mp3')

df.head(6)
df = df[df.Duration >= 1.0]

df = df.reset_index(drop=True)

df
# we plot two seconds

plot_sigx2(d1[df.Start[0]:df.Start[0]+2*SR], d1[df.Start[1]:df.Start[1]+2*SR], 

           name1='btbwar 1', name2='btbwar 2')
mask = get_mask(v2, prom=0.3, dur=0.3, sr=SR)

df = get_regions(mask, SR, 'btnwar', 'XC135117.mp3')

df = df[df.Duration >= 1.0]

df = df.reset_index(drop=True)

df
plot_sigx2(d2[df.Start[0]:df.Start[0]+2*SR], d2[df.Start[1]:df.Start[1]+2*SR], 

           name1='btnwar 1', name2='btnwar 2')
# free up memory

%xdel d1

%xdel d2

%xdel cs1

%xdel cs2
def img_resize(cs, w=512, h=512, log=True, lthres=-30):

    buf = io.BytesIO()

    if log == True:

        plt.imsave(buf, 20*np.log10(np.abs(cs)), cmap=CMAP, format='png', vmax=0, vmin=lthres)

    else:

        plt.imsave(buf, np.abs(cs), cmap=CMAP, format='png')

    buf.seek(0)

    img_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)

    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)



# Parameters:

#   filename: mp3-file

#   voices: # of scales per octave

#   sr: sampling frequency (Hz)

#   low_freq: low freq cutoff (Hz)

#   thres: scaleogram threshold (dB)

#   prom: peak detect prominence

#   peakdur: peak extension (s)

#   sigthres: smallest signature detection to process (s)

#   siglen: length of output signature (s)

#   img_size: output image size

#   outdir: output directory

def scaleo_extract(filename, voices=12, sr=22050, low_freq=40, thres=-30, prom=0.3, 

                   peakdur=0.3, sigthres=1, siglen=2, img_size=512, outdir='.'):

    d = rd_file(filename)

    cs, _ = cwt2(d, nv=voices, sr=sr, low_freq=low_freq) # wavelet transform

    v = calc_var(cs, thres) # coefficient variance

    peaks, _ = find_peaks(v, prominence=prom) 

    m = mask_sig(len(v), peaks, sr=sr, dur=peakdur) # create signal mask

    df = get_regions(m, sr, filename.split('/')[-2], filename.split('/')[-1])

    df = df[df.Duration >= sigthres] # filter out insignificant signatures

    df = df.reset_index(drop=True)

    if len(df) > 0:

        for i in range(len(df)):

            img = img_resize(cs[:,df.Start[i]:df.Start[i]+siglen*sr], 

                             w=img_size, h=img_size, log=True, lthres=thres)

            fn = df.Species[i]+'-'+filename.split('/')[-1].split('.')[-2]+"-{:03d}.jpg".format(i) 

            cv2.imwrite(outdir+'/'+fn, img)

    return df
%mkdir ./tmp
flist = ['/btbwar/XC139608.mp3', '/btbwar/XC51863.mp3', '/btbwar/XC134502.mp3', '/btbwar/XC415596.mp3']

for i in flist:

    scaleo_extract(TRAIN_DIR+i, outdir='./tmp')
import glob



images = glob.glob('./tmp' + "/btbwar*.jpg")

plt.figure(figsize=(20,20))

columns = 4

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.imshow(mpimg.imread(image))

    plt.axis('off')
flist = ['/btnwar/XC135117.mp3', '/btnwar/XC135495.mp3', '/btnwar/XC173264.mp3', '/btnwar/XC501261.mp3', '/btnwar/XC486860.mp3']

for i in flist:

    scaleo_extract(TRAIN_DIR+i, outdir='./tmp')
images = glob.glob('./tmp' + "/btnwar*.jpg")

plt.figure(figsize=(20,25))

columns = 5

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.imshow(mpimg.imread(image))

    plt.axis('off')
!rm -fr tmp