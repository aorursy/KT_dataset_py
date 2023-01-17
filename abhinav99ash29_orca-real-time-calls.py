# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from glob import glob

import pandas as pd

import numpy as np

import os

df = pd.read_table('/kaggle/input/dataset/Round2_OS_07_05/train.tsv')

df.groupby('wav_filename')

df = df.set_index('wav_filename')

df.sort_values(by=['start_time_s'], inplace=True)

df.head()

files = list(set(df.index.values.tolist()))

all_files = [os.path.basename(x) for x in glob('/kaggle/input/dataset/Round2_OS_07_05/wav/*.wav')]

## Utils



def audio_norm(data):

    '''Normalization of audio'''

    max_data = np.max(data)

    min_data = np.min(data)

    data = (data - min_data) / (max_data - min_data + 1e-6)

    return data - 0.5



def padding(data, input_length):

    '''Padding of samples to make them of same length'''

    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset:(input_length + offset)]

    else:

        if input_length > len(data):

            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)

        else:

            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    return data
from scipy.io import wavfile

positive = []

negative = []

for file in files:

    prev = 0

    fs, data = wavfile.read('/kaggle/input/dataset/Round2_OS_07_05/wav/1562344334_0004.wav')

    ## Preprocess

    data = audio_norm(data)

    print(file)

    for x in np.array(df.loc[file]).reshape(1,-1):

        print(x)

        negative.append(data[int(prev*20000) : int(x[0]*20000)])

        positive.append(data[int(x[0]*20000) : int(x[1]*20000)])

        prev = x[0]

        

    

        
no_call = np.concatenate(negative, axis=0)

call = np.concatenate(positive, axis=0)
import matplotlib.pyplot as plt

plt.plot(call)

print(len(call))
import librosa

chunk_size = 20000

data = call.copy()

call_lpc = []

while len(data) >= chunk_size:

        chunk = data[:chunk_size]

        call_lpc.append(librosa.core.lpc(chunk,39))

        data = data[chunk_size:]
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

call_lpc = np.array(call_lpc)

call_lpc = StandardScaler().fit_transform(call_lpc)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(call_lpc)

principalComponents = np.array(principalComponents)

plt.plot(principalComponents[:,0],principalComponents[:,1],'.')