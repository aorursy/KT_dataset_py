# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/training/training.csv')
df.head(1)
images = df['Image'].copy()
features = df.drop(['Image'], axis=1).copy()
Images_df = pd.DataFrame(np.zeros((images.shape[0],96*96)))
Images_ = np.zeros((images.shape[0],96*96))
for i, img in enumerate(images):
    Images_[i] = [float(x) for x in img.split(' ')]
import matplotlib.pyplot as plt
x0 = features.iloc[0,0::2].values
y0 = features.iloc[0,1::2].values
plt.imshow(Images_[0].reshape(96,96))
plt.scatter(x0, y0, c='red',s = 5)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(96,96,1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))

model.add(Flattena())
model.add(Dense(512))
