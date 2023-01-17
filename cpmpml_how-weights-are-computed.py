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

from matplotlib import pyplot as plt
%matplotlib inline
event_id = 3
hits = pd.read_csv('../input/train_1/event00000100%d-hits.csv' % event_id)
particles = pd.read_csv('../input/train_1/event00000100%d-particles.csv' % event_id)
truth = pd.read_csv('../input/train_1/event00000100%d-truth.csv' % event_id)

hits = hits.merge(truth, how='left', on='hit_id')
hits = hits.merge(particles, how='left', on='particle_id')
hits['dv'] = np.sqrt((hits.vx - hits.tx) ** 2 + \
                     (hits.vy - hits.ty) ** 2 + \
                     (hits.vz - hits.tz) ** 2)
hits = hits.sort_values(by=['particle_id', 'dv']).reset_index(drop=True)

hits['rank'] = hits.groupby('particle_id').cumcount()
hits['len'] = hits.groupby('particle_id').particle_id.transform('count')
hits['rank'] = (hits['rank']) / (hits['len'] - 1)
hits['weight'] /= hits.groupby('particle_id').weight.transform('max')
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

ax.scatter(hits['rank'], hits['weight'], alpha=0.1, marker='+')