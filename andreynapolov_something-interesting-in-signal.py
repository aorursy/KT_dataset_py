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

from numpy.fft import fft
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/remove-drift-using-a-sine-function/train_wo_drift.csv')[2_000_000:2_500_000]

dft = fft(df.signal)
freq = (np.abs(dft[:len(dft)]))

for i in range(len(freq)):
    if freq[i] < 30_000: 
        freq[i] = 0

rsig = np.abs(np.fft.ifft(freq))
plt.plot(rsig)
plt.show()