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
import soundfile as sf

import IPython.display as ipd

import matplotlib.pyplot as plt
sound, samplerate = sf.read("/kaggle/input/Childish-Gambino-Me-and-Your-Mama-_Official-Audio_.wav")
sound.shape
sound_ch1 = sound[:,0]
eixo_x = range(10)

eixo_y = range(10,20)

plt.plot(eixo_x, eixo_y)

plt.show()
eixo_y = [i**2 for i in range(10)]

plt.plot(eixo_y)

plt.show()
plt.plot(sound)

plt.show()
plt.plot(sound_ch1)

plt.show()
# ipd.Audio("/kaggle/input/Childish Gambino - Me and Your Mama (Official Audio).mp3")
# ipd.Audio(sound_ch1, rate=samplerate)
peak_max = sound_ch1.max()

peak_min = sound_ch1.min()

print("Valor máximo:", peak_max)

print("Valor mínimo:", peak_min)
mean_value = sound_ch1.mean()

print("Média:", mean_value)
plt.plot(sound_ch1)

plt.plot(np.repeat(mean_value,sound_ch1.shape[0]) ,label="media")

plt.plot(np.repeat(peak_max,sound_ch1.shape[0]),label="pico maximo")

plt.plot(np.repeat(peak_min,sound_ch1.shape[0]), label="pico minimo" )

plt.show()
def separate_sound(sound, samplerate, x, y):

    return sound[x*samplerate:y*samplerate]

plt.plot(separate_sound(sound_ch1, samplerate, 20, 30))

plt.show()

plt.plot(separate_sound(sound_ch1, samplerate, 70, 90))

plt.show()

plt.plot(separate_sound(sound_ch1, samplerate, 250, 260))

plt.show()
windows_steps = int(sound_ch1.shape[0]/samplerate)

windows = np.zeros(samplerate)

for step in range(windows_steps):

    window = separate_sound(sound_ch1, samplerate, step, step+1)

    windows = np.vstack([windows, window])

windows = windows[1:]
peaks_max = windows.max(axis=1)

means = windows.mean(axis=1)

peaks_min = windows.min(axis=1)

print("Maximos e mínimos")

plt.plot(peaks_max)

plt.plot(peaks_min)

plt.show()

print("Média")

plt.plot(means)

plt.show()

print("Som: ")

plt.plot(sound_ch1)

plt.show()
letras = np.array([chr(i) for i in range(65,91)])

posicoes = np.arange(91-65) + 1

df_letras = pd.DataFrame(data=np.vstack([letras, posicoes]).T, columns=['letras', 'posicoes'])

df_letras.head()


seconds = np.arange(windows.shape[0])

data = np.vstack([seconds,peaks_max,peaks_min]).T

df = pd.DataFrame(data=data,columns=['segundo','maximo','minimo'])

df.head()