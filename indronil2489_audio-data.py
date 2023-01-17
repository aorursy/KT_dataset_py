# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train= pd.read_csv("/kaggle/input/birdsong-recognition/train.csv")
df_test= pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")
f="../input/birdsong-recognition/train_audio/"
path_train=[]
y_train=[]
for i in range(0,len(df_train)):
    f="../input/birdsong-recognition/train_audio/"
    f=f+df_train.iloc[i,2]+'/'+df_train.iloc[i,7]
    path_train.append(f)
    y_train.append(df_train.iloc[i,2])
len(path_train)
from pydub import AudioSegment

sound = AudioSegment.from_mp3(path_train[i])

raw_data = sound.raw_data
data = np.fromstring(raw_data, dtype=np.int16)
sample_rate = sound.frame_rate
sample_size = sound.sample_width
channels = sound.channels
x_train=[]
from pydub import AudioSegment

for i in range(0, len(path_train)):
    sound = AudioSegment.from_mp3(path_train[i])
    raw_data = sound.raw_data
    data = np.fromstring(raw_data, dtype=np.int16)
    x_train.append(data)
#sample_rate = sound.frame_rate
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(data)
plt.savefig("./my_img.png")

import IPython
import librosa
x, sr = librosa.load(path=f, mono=True)
IPython.display.Audio(data=data, rate=sr)
data.shape