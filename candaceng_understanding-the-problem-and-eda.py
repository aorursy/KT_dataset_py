import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')
test = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/test.csv')
print(train.shape)
print(test.shape)
sample_start_23 = train.loc[23, train.columns.str.startswith('start')]
sample_stop_23 = train.loc[23, train.columns.str.startswith('stop')]
print(sample_start_23)
print(sample_stop_23)
sample_start_23 = np.asarray(sample_start_23).reshape(25, 25)
sample_stop_23 = np.asarray(sample_stop_23).reshape(25, 25)
train.loc[23, 'delta']
fig, ax = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
ax[0] = plt.subplot2grid((1,2), (0,0), colspan=1)
ax[1] = plt.subplot2grid((1,2), (0,1), colspan=1)
ax[0].imshow(sample_start_23)
ax[0].set_title('Start Board 23')
ax[1].imshow(sample_stop_23)
ax[1].set_title(f'Board after 1 time step')
plt.show()