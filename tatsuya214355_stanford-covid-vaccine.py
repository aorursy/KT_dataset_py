import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import json

import ast

import seaborn as sns

import os



from itertools import cycle

pd.set_option('max_columns', None)



!ls -GFlash --color ../input/stanford-covid-vaccine/

!ls -GFlash --color ../input/stanford-covid-vaccine/bpps/ | head

!du -h ../input/stanford-covid-vaccine/bpps/
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

train.shape, test.shape, ss.shape
train.head()
test.head()
ss.head()
bpps_files = os.listdir('../input/stanford-covid-vaccine/bpps/')

example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_files[0]}')

print('bpps file shape:', example_bpps.shape)
for i in range(100):

    example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_files[i]}')

    print('bpps file shape:', example_bpps.shape)
# 1行目の7列目の要素と7行目の1列目の要素が一致していることを確認

example_bpps[0], example_bpps[6]
plt.style.use('default')

fig, axs = plt.subplots(5, 5, figsize=(15, 15))

axs = axs.flatten()

for i, f in enumerate(bpps_files):

    if i == 25:

        break

    example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{f}')

    axs[i].imshow(example_bpps)

    axs[i].set_title(f)

plt.tight_layout()

plt.show()
ss.to_csv('submission.csv', index=False)