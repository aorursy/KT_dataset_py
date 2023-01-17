# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_json('../input/train.json')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load the training dataset.

train = pd.read_json('../input/train.json')
# Isolation function.

def iso(arr):

    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))

    return p * np.reshape(np.array(arr), [75,75])

# Size in number of pixels of every isolated object.

def size(arr):     

    return np.sum(arr<-5)
# Feature engineering iso1 and iso2.

train['iso1'] = train.iloc[:, 0].apply(iso)

train['iso2'] = train.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.

train['s1'] = train.iloc[:,5].apply(size)

train['s2'] = train.iloc[:,6].apply(size)

train['size'] = train.s1+train.s2
# How works s1 on the discrimination

print(train.groupby('is_iceberg')['s1'].mean())
# Hist comparison

train.groupby('is_iceberg')['s1'].hist(bins=60, alpha=.6)
# How works s2 on the discrimination

print(train.groupby('is_iceberg')['s2'].mean())
# Hist comparison

train.groupby('is_iceberg')['s2'].hist(bins=60, alpha=.6)
# How works size on the discrimination

print(train.groupby('is_iceberg')['size'].mean())
# Hist comparison

train.groupby('is_iceberg')['size'].hist(bins=60, alpha=.6)
# Indexes for ships or icebergs.

index_ship=np.where(train['is_iceberg']==0)

index_ice=np.where(train['is_iceberg']==1)
# For ploting

def plots(band,index,title):

    plt.figure(figsize=(12,10))

    for i in range(12):

        plt.subplot(3,4,i+1)

        plt.xticks(())

        plt.yticks(())

        plt.xlabel((title))

        plt.imshow(np.reshape(train[band][index[0][i]], (75,75)),cmap='gist_heat')

    plt.show()  
plots('band_1',index_ship,'band1 ship')
plots('band_1',index_ice,'band1 iceberg')
plots('iso1',index_ship,'iso1 ship')
plots('iso1',index_ice,'iso1 iceberg')