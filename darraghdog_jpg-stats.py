# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

from tqdm import tqdm
dfls = []

for t, (dirname, _, filenames) in enumerate(os.walk('/kaggle/input')):

    for tt, filename in enumerate(tqdm(filenames)):

        img = cv2.imread(os.path.join(dirname, filename))

        dfls.append([filename, img.mean(), img.std()])
df = pd.DataFrame(dfls, columns = ['Image', 'Mean', 'Std'])
df.head()

df.to_csv('img_stats.csv.gz', index = False, compression = 'gzip')