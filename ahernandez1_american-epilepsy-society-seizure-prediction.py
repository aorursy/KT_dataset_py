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
import scipy.io
from scipy.signal import spectrogram
from scipy.signal import resample
#Visualizing an example:
interictal_tst = '../input/seizure-prediction/Patient_1/Patient_1/Patient_1_interictal_segment_0001.mat'
preictal_tst = '../input/seizure-prediction/Patient_1/Patient_1/Patient_1_preictal_segment_0001.mat'
interictal_data = scipy.io.loadmat(interictal_tst)
preictal_data = scipy.io.loadmat(preictal_tst)
interictal_data.get('interictal_segment_1')
interictal_data['interictal_segment_1'][0][0][1]
a = interictal_data['interictal_segment_1'][0][0][1]
b = interictal_data['interictal_segment_1'][0][0][2]
a
a = np.array(interictal_data['interictal_segment_1'][0][0][2], dtype=np.uint64)
b = np.array(interictal_data['interictal_segment_1'][0][0][1], dtype=np.uint64)
(a[0][0]) * (b[0][0])
import matplotlib.pyplot as plt
import numpy as np

w=10
h=10
fig = plt.figure(figsize=(8,8))
columns = 4
rows = 5

for i in range(1, columns * rows +1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


columns = 1
rows = interictal_data['interictal_segment_1'][0][0][0].shape[0]
fig = plt.figure(figsize=(40,40))
for plot in range(1, rows +1):
    img = [interictal_data['interictal_segment_1'][0][0][0][plot-1]]
    fig.add_subplot(rows, columns, plot)
    plt.imshow(img)
plt.show()
import matplotlib.pyplot as plt

for plot in range(interictal_data['interictal_segment_1'][0][0][0].shape[0]):
    plt.figure(figsize=(30,1.2))
    plt.plot(interictal_data['interictal_segment_1'][0][0][0][plot])
    plt.ylabel(interictal_data['interictal_segment_1'][0][0][3][0][plot])
    plt.show()
import matplotlib.pyplot as plt

for plot in range(preictal_data['preictal_segment_1'][0][0][0].shape[0]):
    plt.figure(figsize=(30,1.2))
    plt.plot(preictal_data['preictal_segment_1'][0][0][0][plot])
    plt.ylabel(preictal_data['preictal_segment_1'][0][0][3][0][plot])
    plt.show()
