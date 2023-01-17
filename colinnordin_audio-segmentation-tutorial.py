from pathlib import Path



import numpy as np

import matplotlib.pyplot as plt

import librosa

import pandas as pd

import IPython.display as ipd
BASE_DIR = Path('../input/birdsong-recognition')

train_df = pd.read_csv(BASE_DIR / 'train.csv')

random_row = train_df.sample().squeeze()
sample_rate = 32000

fpath = BASE_DIR / 'train_audio' / random_row['ebird_code'] / random_row['filename']

audio, _ = librosa.core.load(fpath, sr=sample_rate, mono=True)



plt.plot(audio)

ipd.display(ipd.Audio(audio, rate=sample_rate))
class SNRSegmenter(object):



    def __init__(self, sample_rate, segment_len_ms, hop_len_ms, noise_len_ms, call_snr):

        self.segment_len_samples = int(sample_rate * segment_len_ms / 1000)

        self.hop_len_samples = int(sample_rate * hop_len_ms / 1000)

        self.noise_len_samples = int(sample_rate * noise_len_ms / 1000)



        self.call_snr = call_snr



    def _get_noise_level(self, sample):

        abs_max = []

        

        if len(sample) > self.noise_len_samples:

            idx = 0

            while idx + self.noise_len_samples < len(sample):

                abs_max.append(np.max(np.abs(sample[idx:(idx+self.noise_len_samples)])))

                idx += self.noise_len_samples

        else:

            abs_max.append(np.max(np.abs(sample)))



        return min(abs_max)



    def __call__(self, sample):

        

        noise_level = self._get_noise_level(sample)



        call_segments = []

        call_snrs = []



        if len(sample) > self.segment_len_samples:

            idx = 0

            while idx + self.segment_len_samples < len(sample):

                segment = sample[idx:(idx+self.segment_len_samples)]

                seg_abs_max = np.max(np.abs(segment))

                if seg_abs_max / noise_level > self.call_snr:

                    call_segments.append(segment)

                    call_snrs.append(seg_abs_max / noise_level)



                idx += self.hop_len_samples



        return call_segments, call_snrs
segment_len_ms = 2500

hop_len_ms = 1000

noise_len_ms = 500

call_snr_thresh = 5



segmenter = SNRSegmenter(sample_rate, segment_len_ms, hop_len_ms, noise_len_ms, call_snr_thresh)



calls, call_snrs = segmenter(audio)
plt.title(f'SNR = {call_snrs[0]}')

plt.plot(calls[0])

ipd.display(ipd.Audio(calls[0], rate=sample_rate))  
plt.title(f'SNR = {call_snrs[5]}')

plt.plot(calls[5])

ipd.display(ipd.Audio(calls[5], rate=sample_rate))