import torch

import torchaudio

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import librosa

import librosa.display



import IPython.display as ipd
# filename = "../input/birdsong-recognition/train_audio/nutwoo/XC462016.mp3"

filename = '../input/birdsong-recognition/train_audio/balori/XC101614.mp3'



waveform, sample_rate = torchaudio.load(filename)



print("Shape of waveform {}".format(waveform.size()))

print("Sample rate of wavefor {}".format(sample_rate))



plt.figure(figsize=(14,5))

plt.plot(waveform.t()) # transpose

ipd.Audio(waveform, rate=sample_rate)
specgram = torchaudio.transforms.Spectrogram()(waveform)



print("Shape of Spectrogram {}".format(specgram.size()))



plt.figure(figsize=(14,5))

plt.imshow(specgram.log2()[0,:,:1200].numpy(), cmap='gray')
specgram = torchaudio.transforms.MelSpectrogram()(waveform)



print("Shape of MelSpectrogram {}".format(specgram.size()))



plt.figure(figsize=(14,5))

plt.imshow(specgram.log2()[0,:,:1000].numpy(), cmap='gray')
new_sample_rate = sample_rate / 10



channel = 0



resampled = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))



print("Shape of resampled waveform: {}".format(resampled.size()))



plt.figure(figsize=(14,5))

plt.plot(resampled[0,:].numpy())
ipd.Audio(resampled, rate=new_sample_rate)
print("Min. of waveform {} \n Max. of waveform {} \n Mean of waveform {}".format(waveform.min(), waveform.max(), waveform.mean()))
def normalize(signal):

    signal_minusmean = signal - signal.mean()

    return signal_minusmean / signal_minusmean.abs().max()



#Normalizing waveform

# print("After normalizing waveform...")

# print("Min. of waveform {}".format(normalize(waveform).min().item()))

# print("Max. of waveform {}".format(normalize(waveform).max().item()))

# print("Mean. of waveform {}".format(normalize(waveform).mean().item()))
# Applying Mu Law encoding

encoded = torchaudio.transforms.MuLawEncoding()(waveform)



print("Shape of encoded waveform {}".format(encoded.size()))



plt.figure(figsize=(14,5))

plt.plot(encoded[0,:].numpy())
ipd.Audio(encoded, rate=sample_rate)
reconstructed = torchaudio.transforms.MuLawDecoding()(encoded)



print("Shape of recovered waveform {}".format(reconstructed.size()))



plt.figure(figsize=(14,5))

plt.plot(reconstructed[0,:].numpy())
ipd.Audio(reconstructed, rate=sample_rate)
err = ((waveform-reconstructed).abs() / waveform.abs()).median()



print("Median error difference between original waveform and its reconstructed version is {:.2%}".format(err))
mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)



print("Shape of transformed waveform: {}".format(mu_law_encoding_waveform.size()))



plt.figure(figsize=(14,5))

plt.plot(mu_law_encoding_waveform[0,:].numpy())
computed = torchaudio.functional.compute_deltas(specgram.contiguous(), win_length=3)



print("Shape of Computed deltas {}".format(computed.size()))



plt.figure(figsize=(14,5))

plt.imshow(computed.log2()[0,:,:1000].numpy(), cmap='gray')
gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)



print("Min. of gain_waveform {} \nMax. of gain_waveform {} \nMean of gain_waveform {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))
ipd.Audio(gain_waveform, rate=sample_rate)
dither_waveform = torchaudio.functional.dither(waveform)

print("Min of dither_waveform: {}\nMax of dither_waveform: {}\nMean of dither_waveform: {}".format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))
ipd.Audio(dither_waveform, rate=sample_rate)
lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)



print("Min. of lowpass_waveform: {}\nMax. of lowpass_waveform: {}\nMean of lowpass_waveform: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))



plt.figure(figsize=(14,5))

plt.plot(lowpass_waveform.t().numpy())
ipd.Audio(lowpass_waveform, rate=sample_rate)
highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=2000)



print("Min of highpass_waveform: {}\nMax of highpass_waveform: {}\nMean of highpass_waveform: {}".format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))



plt.figure(figsize=(14,5))

plt.plot(highpass_waveform.t().numpy())
ipd.Audio(highpass_waveform, rate=sample_rate)