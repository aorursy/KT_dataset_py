import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from scipy import signal


train_df = pd.read_json("../input/train.json")



# Train data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
# Edge Detection kernel

edge = np.array([[-100,-100,-100],[-100,100,-100],[-100,-100,-100]])

arr = signal.convolve2d(np.reshape(np.array(x_band1[5]),(75,75)),edge,mode='valid')

plt.imshow(arr)

plt.show()



# Sharpen kernel

sharpen = np.array([[0,-1,0],[-1,-5,-1],[0,-1,0]])

arr = signal.convolve2d(np.reshape(np.array(x_band1[5]),(75,75)),sharpen,mode='valid')

plt.imshow(arr)

plt.show()