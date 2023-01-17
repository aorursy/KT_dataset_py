# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json, sys, random
import numpy as np
# download dataset from json object
f = open(r'../input/ships-in-satellite-imagery/shipsnet.json')
dataset = json.load(f)
f.close()
data = np.array(dataset['data']).astype('uint8')
labels = np.array(dataset['labels']).astype('uint8')
data.shape
print(labels.shape)
print(labels[0:10])
n_spectrum = 3 # color chanel (RGB)
width = 80
height = 80
X = data.reshape([-1, n_spectrum, width, height])

X.shape
X[0,1]
data
ship = np.where(labels==1)[0][0:20]
noship = np.where(labels==0)[0][0:20]
ship,noship
import matplotlib.pyplot as plt
p,axes = plt.subplots(5,4,sharex=True,sharey=True)
for i in ship:
    ax = axes[int(i/4)][int(i%4)]
    ax.imshow(np.moveaxis(X[i],0,-1))
p,axes = plt.subplots(5,4,sharex=True,sharey=True)
count = 0
for i in noship:
    ax = axes[int(count/4)][int(count%4)]
    ax.imshow(np.moveaxis(X[i],0,-1))
    count += 1
# from PIL import Image
# Image.fromarray(X[0,1])
# X[0,2]
df_m = np.mean(X[:,:,:],axis=-1).mean(axis=-1)
df_m.shape
np.mean(X[:,:,:],axis=-1)
np.mean(X[:,:,:],axis=-1).mean(axis=-1)
df_m = pd.DataFrame(df_m,columns = ['R','G','B'])
df_m
df_m.hist()
# Your code here...
