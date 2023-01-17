%matplotlib inline
!pip install SoundFile
import tensorflow as tf

import os

from tensorflow.contrib.framework.python.ops import audio_ops

import numpy as np

import soundfile as sf

import matplotlib.pyplot as plt

import re

from scipy import signal

from scipy import misc

from scipy import stats

import pandas as pd
tf.enable_eager_execution() 

#tf.disable_eager_execution()
def LoadToSpectrogram(wav_file):

    """Load audio file, return spectrogram,rate,and waveform

    

    accepts either mp3 or flac. Not tested on other formats.

    

    """

    proot,pext = os.path.splitext(wav_file)

    if "mp3" in pext.lower():

        song_binary = tf.read_file(wav_file) # expecting all my mp3 files to fit this format, eg only from Cornell database

        song_waveform = tf.contrib.ffmpeg.decode_audio(song_binary,file_format='mp3',samples_per_second=48000,channel_count=1)

        np_song_waveform = song_waveform.numpy()[0]

        rate = 48000

    else:    

        np_song_waveform,rate = sf.read(wav_file,always_2d=True,dtype='float32')

        song_waveform = tf.convert_to_tensor(np_song_waveform)

    spectrogram = audio_ops.audio_spectrogram(song_waveform,window_size=1024,stride=64)

    return spectrogram,rate,np_song_waveform
def snip_up_spectrogram(specnum,min_extent = 50):

    """ locate and return features in spectrogram, and their locations in the original 

    

    Arguments

        specnum: a numpy array shape (n,m)

    Keywork Argument

        min_extent: do not return segments shorter than this

        

    The snip_extents can be used to locate the portion of the sound file the 

    spectrogram snip represents. eg, if window_size is 1024 and stride is 64,

    wavsnip = wavnum[64*pair[1]:64*pair[1]+64*pair[2]+1024]

    

    """

    def tmeanmid(a) :

        midmax = np.max(a)/3.5

        return stats.tmean(a,(midmax,None))

    filta,filtb = signal.iirdesign(0.0025,0.5,1,20)

    filtered_maxima = signal.filtfilt(filta,filtb,np.apply_along_axis(tmeanmid,1,specnum))

    maxf = np.max(filtered_maxima)

    gate = filtered_maxima/maxf>0.076

    pillars = np.cumsum(1-gate)*gate

    startextents = np.transpose(np.unique(pillars,return_counts=True,return_index=True))

    snip_extents = []

    snips = []

    for pair in startextents:

        if (pair[0]!=0) and (pair[2]>min_extent):

            snips.append(specnum[pair[1]:pair[1]+pair[2]])

            snip_extents.append(pair)

    return snips,snip_extents   
def MakeGreyFromSnip(snip,crop_len = 512,imsize = 512,imwidth = None,low_log_bound = 1.2):

    """ make a gray scale numpy image from a spectrogram

    

    Arguments

        snip: spectrogram as 2D numpy array

        

    Keyword arguments

        crop_len:  

            If the snip exceeds this length, crop to it. 

            If the snip is shortr than this length, pad with zeros 

        imsize:

            Resize the image to this size in pixels

        imwidth:

            If imwidth is not specified, make the image square, else image is (imsize,imwidth)

        low_log_bound:

            spectrogram is nomalized to 1e4, then converted to log10, to give a range from _ to 4

            the lower bound is then clipped to 'low_log_bound'

            and then shifted to fill the new range

            In the resulting image, 

                0 maps to 10^low_log_bound, 

                1 maps to 10^4

                

    """

    len = np.shape(snip)[0]

    pad_n = crop_len - len

    if pad_n<0:

        pad_n=0

    snipMax = np.max(snip)

    log_norm_snip = np.log10(10000.*snip/snipMax)

    snappedf = np.transpose(np.pad(log_norm_snip[0:crop_len,:],((0,pad_n),(0,0)),mode='constant'))

    snapped = np.maximum(snappedf,low_log_bound) - low_log_bound

    spmax = np.max(snapped)

    snapped_norm_tensor = tf.convert_to_tensor(snapped / spmax)

    sne = tf.expand_dims(tf.expand_dims(snapped_norm_tensor,0),3)

    _imwidth = imsize

    if imwidth is not None:

        _imwidth = imwidth

    resize = tf.image.resize_bilinear(sne,[imsize,_imwidth])

    return resize.numpy()[0,:,:,0]
data,rate = sf.read("../input/songs/songs/xc101862.flac",always_2d=True,dtype='float32')
spectrogram = audio_ops.audio_spectrogram(tf.convert_to_tensor(data),window_size=1024,stride=64)
specnum = spectrogram.numpy()[0]
snips,_ = snip_up_spectrogram(specnum)
grey = MakeGreyFromSnip(snips[1])
plt.imshow(grey,origin='lower')

plt.show()
roi_snips,_ = snip_up_spectrogram(specnum[:,20:300])

roi_grey = MakeGreyFromSnip(roi_snips[1])

plt.imshow(roi_grey,origin='lower')

plt.show()
long_grey = MakeGreyFromSnip(roi_snips[1],crop_len = 1512,imwidth = 1500)

plt.imshow(long_grey,origin='lower')

plt.show()
# some filter parameters, including a convolution kernel adjusted to find chip clusters

filta,filtb = signal.iirdesign(0.004,0.5,1,20)

def tmeanmid(a) :

    midmax = np.max(a)/3.5

    return stats.tmean(a,(midmax,None))

step_kernel = np.zeros(101)

step_kernel_width = 40

step_kernel[:] = -0.1

step_kernel[50-40:50+40] = 1.0

km = np.sum(step_kernel)

step_kernel = step_kernel / km


peaks =  np.apply_along_axis(tmeanmid,1,specnum[:,20:300])

peaks_max = np.max(peaks)

peaks = peaks / peaks_max

plt.plot(peaks)
roi = slice(900,5000)

convolved_maxima = signal.convolve(peaks,step_kernel,mode='same')

plt.plot(convolved_maxima[roi])

plt.show()
filtered_maxima = signal.filtfilt(filta,filtb,convolved_maxima)

plt.plot(filtered_maxima[roi])



peak_indicis,peak_properties = signal.find_peaks(filtered_maxima,width = (4,1000),height=(0.06,10))

my_nearest_peak_value = np.ones(np.shape(peaks)[0])

widths = peak_properties['widths']

for pin,pw in zip(peak_indicis,widths):

    my_nearest_peak_value[pin-int(pw):pin+int(pw)] = filtered_maxima[pin]

    

plt.plot(filtered_maxima[roi]/my_nearest_peak_value[roi])

plt.show()
gate = filtered_maxima/my_nearest_peak_value>0.4

pillars = np.cumsum(1-gate)*gate

plt.plot(gate[roi])

plt.show()

plt.plot(pillars[roi])

plt.show()
startextents = np.transpose(np.unique(pillars,return_counts=True,return_index=True))

print(startextents)

pair = startextents[1]

plt.plot(peaks[pair[1]-10:pair[1]+pair[2]+10])
cluster = MakeGreyFromSnip(specnum[pair[1]-10:pair[1]+pair[2]+10,20:300],crop_len = 1300,imwidth = 1000)

plt.imshow(cluster,origin='lower')
