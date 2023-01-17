# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.fname
fname = os.listdir('../input/severstal-steel-defect-detection/train_images')
fname[:5]
#fname = df.ImageId.to_list()

cname = list(range(1,5))

fname = ['{}_{}'.format(a,b) for b in cname for a in fname]

fname[:5]
df = pd.DataFrame()

df['ImageId'] = fname
df['ClassId'] = df['ImageId'].str.split('_', expand=True)[1]

df['ImageId'] = df['ImageId'].str.split('_', expand=True)[0]
df =  df.sort_values(['ImageId','ClassId']).reset_index(drop=True)
df.head()
tr_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

tr_df.head()
tr_df.dtypes
df.dtypes
df['ClassId'] = df['ClassId'].astype(int)
train = df.merge(tr_df, on=['ImageId','ClassId'], how='left')
train.head()
train.ClassId.value_counts()
train.dtypes
train['ImageId_ClassId'] = train['ImageId']+"_"+train['ClassId'].astype(str)
train = train[['ImageId_ClassId','EncodedPixels']]

train.head()
train.to_csv("train.csv", index=False)