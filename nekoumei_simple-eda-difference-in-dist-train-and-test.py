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

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
cols = [f'var_{i}' for i in range(200)]
width_num = 5

height_num = 40

fig = plt.figure(figsize=[4*width_num,3*height_num])

for i in tqdm_notebook(range(200)):

    ax = fig.add_subplot(height_num,width_num,i+1)

    _ = ax.hist(train[cols[i]], bins=100, alpha=0.5, density=True, label='train')

    _ = ax.hist(test[cols[i]], bins=100,alpha=0.5, density=True, label='test')

    _ = ax.set_title(cols[i])

    ax.legend()