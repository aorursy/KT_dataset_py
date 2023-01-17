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
import glob

from PIL import Image

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv') #ImageId_ClassId, EncodedPixels

train['class'] = train['ImageId_ClassId'].map(lambda x: x.split('_')[1])

train['path'] = train['ImageId_ClassId'].map(lambda x: '../input/train_images/' + x.split('_')[0])

train.head()
train_df = pd.DataFrame(glob.glob('../input/test_images/**'), columns=['path'])

train_df['ImageId'] = train_df['path'].map(lambda x: x.split('/')[-1])

train_df.head()
test = pd.DataFrame(glob.glob('../input/test_images/**'), columns=['path'])

test['ImageId'] = test['path'].map(lambda x: x.split('/')[-1])

test.head()
plt.imshow(Image.open(test.path[0]))
plt.imshow(Image.open(train.path[0]))
train.columns.values
sub = []

for i in range(5):

    #train here

    trainTemp = train[train['class'] == i+1].reset_index(drop=True).copy()

    

    #predict here

    subTemp = test.copy()

    subTemp['ImageId_ClassId'] = subTemp.apply(lambda r: '_'.join([r['ImageId'], str(i+1)]), axis=1)

    subTemp['EncodedPixels'] = None

    sub.append(subTemp.copy())

sub = pd.concat(sub)

sub[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)
df = pd.read_csv('submission.csv')

df.count()