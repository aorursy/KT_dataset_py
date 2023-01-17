from math import sqrt

import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')

data.head()
data.info()
data.emotion.value_counts()
num_classes = 7

width = 48

height = 48

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

classes=np.array(emotion_labels)
depth = 1

height = int(sqrt(len(data.pixels[0].split()))) 

width = int(height)

print(height, width)
h, w = 10, 10        

nrows, ncols = 1, 8  # array of sub-plots

figsize = [20, 30]     # figure size, inches





# create figure (fig), and array of axes (ax)

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)



# plot simple raster image on each sub-plot

for i, axi in enumerate(ax.flat):

    # i runs from 0 to (nrows*ncols-1)

    # axi is equivalent with ax[rowid][colid]

    img = np.mat(data.pixels[i]).reshape(height, width) 

    axi.imshow(img, cmap = plt.cm.gray)

    # get indices of row/column

    rowid = i // ncols

    colid = i % ncols

    axi.set_title(emotion_labels[data.emotion[i]])



plt.tight_layout(True)

plt.show()