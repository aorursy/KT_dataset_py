import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../input/'))
from utils import ESC50

train_splits = [1,2,3,4]
test_split = 5

shared_params = {'csv_path': '../input/esc50.csv',
                 'wav_dir': '../input/audio/audio',
                 'dest_dir': '../input/audio/audio/16000',
                 'audio_rate': 16000,
                 'only_ESC10': True,
                 'pad': 0,
                 'normalize': True}

train_gen = ESC50(folds=train_splits,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=True,
                  **shared_params).batch_gen(16)

test_gen = ESC50(folds=[test_split],
                 randomize=False,
                 strongAugment=False,
                 random_crop=False,
                 inputLength=4,
                 mix=False,
                 **shared_params).batch_gen(16)

X, Y = next(train_gen)
X.shape, Y.shape
df = pd.DataFrame.from_csv('../input/esc50.csv')
classes = df[['target', 'category']].as_matrix().tolist()
classes = set(['{} {}'.format(c[0], c[1]) for c in classes])
classes = np.array([c.split(' ') for c in classes])
classes = {k: v for k, v in classes}
import scipy
from scipy import signal
import IPython.display as ipd

fig, axs = plt.subplots(2, 5, figsize=(13, 4))
for idx in range(5):
    i, j = int(idx / 5), int(idx % 5)
    x = X[idx]
    sampleFreqs, segmentTimes, sxx = signal.spectrogram(x[:, 0], 16000)
    axs[i*2][j].pcolormesh((len(segmentTimes) * segmentTimes / segmentTimes[-1]),
                         sampleFreqs,
                         10 * np.log10(sxx + 1e-15))
    #axs[i*2][j].set_title(classes[seen_classes[idx]])
    axs[i*2][j].set_axis_off()
    axs[i*2+1][j].plot(x)
    #axs[i*2+1][j].set_axis_off()
    
plt.show()

for idx in range(5):
    x = X[idx]
    ipd.display(ipd.Audio(x[:, 0], rate=16000))

