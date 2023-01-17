# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mne.preprocessing import read_ica, ICA

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls /kaggle/input/rawsbeforehoes/
import matplotlib.pyplot as plt



import mne

from mne import read_proj

from mne.io import read_raw_fif



from mne.datasets import sample

import os
path = "/kaggle/input/rawsbeforehoes/P01-raw.fif"

raw = mne.io.read_raw_fif(path,preload=True)
raw.info
raw.plot()
ica = read_ica('/kaggle/input/openmiir/eeg/preprocessing/ica/P01-100p_64c-ica.fif')

ica.plot_components()

ica.plot_sources(raw)
reconst_raw = raw.copy()

s = ica.apply(reconst_raw)
X = raw.get_data()

X_clean = s.get_data()
sfreq = raw.info['sfreq']
from pylab import *

plot(X[1, 0:int(10 * sfreq)])
raw.plot(n_channels = 2, scalings = 'auto')
plot(X_clean[0])
raw.plot_psd()
reconst_raw.plot_psd()
plot(X_clean[0])
reconst_raw.plot(duration=20.0, start=0.0, n_channels=1)
# show some frontal channels to clearly illustrate the artifact removal

chs = raw.ch_names

chan_idxs = [raw.ch_names.index(ch) for ch in chs]

raw.plot(order=chan_idxs, start=12, duration=4)

reconst_raw.plot(order=chan_idxs, start=12, duration=4)