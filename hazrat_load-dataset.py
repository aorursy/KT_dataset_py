import numpy as np 

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



urdu_dataset = np.load('../input/uhat-urdu-handwritten-text-dataset/uhat_dataset.npz')



x_train = urdu_dataset['x_chars_train']

y_train = urdu_dataset['y_chars_train']
print('Size of train data for characters: ',x_train.shape)

print('Size of train labels for characters: ',y_train.shape)
