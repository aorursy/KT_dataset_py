# import os
# os.walk('/kaggle/input')

import numpy as np
import matplotlib.pyplot as plt

folder = '/kaggle/input/2020-01-deep/'
mel = np.load( folder + 'train/0.npy')
plt.imshow(mel)
mel.shape
import pandas as pd

y_train = pd.read_csv(folder + 'y_train.csv')
y_train
sample = pd.read_csv(folder + 'sample_submission.csv')
sample
