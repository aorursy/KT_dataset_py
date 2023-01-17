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
train_df = pd.read_csv('../input/train_relationships.csv')
train_df.head()
import matplotlib.pyplot as plt

from PIL import Image

from pathlib import Path
img_path = Path('../input/train/')
img_list = os.listdir(img_path / train_df.p1[0])
fig,ax = plt.subplots(2,5, figsize=(50,20))



for i in range(len(img_list)):

    with open(img_path / train_df.p1[0] / img_list[i] ,'rb') as f:

        img = Image.open(f)

        ax[i%2][i//2].imshow(img)

fig.show()
img_list = os.listdir(img_path / train_df.p2[0])
fig,ax = plt.subplots(2,5, figsize=(50,20))



for i in range(len(img_list)):

    with open(img_path / train_df.p2[0] / img_list[i] ,'rb') as f:

        img = Image.open(f)

        ax[i%2][i//2].imshow(img)

fig.show()