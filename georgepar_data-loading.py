# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import os

# Any results you write to the current directory are saved as output.
# load a spectrogram
import gzip

SAMPLE =  '../input/fma_small_spectrograms/fma_small_spectrograms/Blues/1042.fused.full.npy.gz'

with gzip.GzipFile(SAMPLE, 'r') as f:
    s = np.load(f)

mel = s[0:128]
chroma = s[128:]
print('Mel Spectrogram shape')

print('(n_features, timesteps)')
print(mel.shape)
print('Chromagram shape')

print('(n_features, timesteps)')
print(chroma.shape)
# Visualize with librosa.display.spec... Leaving for exercise