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
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
SAMPLE_SIZE_PER_CSV = 35000
PIXELS = 64
NCSVS = 100
ALL_TRAIN_CSV = os.listdir(os.path.join("../input/train_simplified/"))
categories = [word.split('.')[0] for word in ALL_TRAIN_CSV]
for y, cat in tqdm(enumerate(categories)):
    df = pd.read_csv(os.path.join("../input/train_simplified/", cat + '.csv'),nrows=SAMPLE_SIZE_PER_CSV)
    df = df[df['recognized']==True][['drawing','word']]
    df['y'] = y
    rnd_index = list(np.arange(NCSVS))*(int(len(df)/NCSVS)+1)
    df['csv_index'] = rnd_index[:len(df)]
    for k in range(NCSVS):
        filename = 'train_k{}.csv'.format(k)
        chunk = df[df.csv_index == k]
        chunk = chunk.drop(['csv_index'], axis=1)
        if y == 0:
            chunk.to_csv(filename, index=False)
        else:
            chunk.to_csv(filename, mode='a', header=False, index=False)
for k in tqdm(range(NCSVS)):
    filename = 'train_k{}.csv'.format(k)
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = shuffle(df)
        df.to_csv(filename + '.gz', compression='gzip', index=False)
        os.remove(filename)

