# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Custom Imports

from subprocess import check_output
from scipy.io import wavfile
# processing silence files
train_audio_path = '../input/train/audio'

print(check_output(["ls", "../input/train/audio/_background_noise_"]).decode("utf8"))

def wavread(file):
    sample_rate, samples = wavfile.read(str(train_audio_path) + '/' + file)
    return np.array(samples)

files = os.listdir(train_audio_path + '/_background_noise_')

tot_files = 0

for f in files:
    if not f.endswith('wav'):
        continue

    f_samples = wavread('/_background_noise_/'+f)
    f_len = len(f_samples)
    f_name = os.path.splitext(f)[0]

    i = 0
    
    while i + 16000 <= f_len and i < int(82):
        wavfile.write(f_name + '_' + str(i) + '.wav', 16000, f_samples[i*16000:i*16000+16000])
        i = i + 1
        tot_files = tot_files + 1

print("Silence files saved:",tot_files)
