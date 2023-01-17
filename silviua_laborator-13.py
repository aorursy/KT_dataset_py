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



import matplotlib.pyplot as plt
import pandas as pd

Signal_col = pd.read_csv("../input/Signal_col.csv")
Signal_col = np.asarray(Signal_col)

print(Signal_col.shape)



plt.plot(Signal_col)
N = 256

kc = int(N/16)

HK = np.zeros(N)

HK[0:kc-1] = 1

HK[N-kc+1:N] = 1

fx = np.linspace(0, 1, N)

plt.plot(fx, HK)
hk = np.fft.ifftshift(np.fft.ifft(HK))

plt.plot(np.arange(N),hk)
L = 65

w = np.transpose(np.blackman(L))

hw = hk[(int)(N/2-np.floor(L/2)) : (int)(N/2+1+np.floor(L/2))] * w



plt.xlabel("Sample index")

plt.title("Windowed filter sequence h(k) using blackman(n)")

plt.plot(np.arange(L), hw)
Signal_col = Signal_col.reshape(399, )

signal_lowpassed = np.convolve(Signal_col, hw)

plt.plot(signal_lowpassed)