# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL

import matplotlib.pyplot as plt

import glob



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if i > 20:

            break

        i = i+1



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
im = PIL.Image.open('/kaggle/input/ships-in-satellite-imagery/shipsnet/shipsnet/1__20170909_181729_0e0f__-122.35067750648894_37.78126618441992.png')
plt.imshow(im);
r, g, b = im.split()



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))



axes[0].imshow(r, cmap="Reds");

axes[1].imshow(g, cmap="Greens");

axes[2].imshow(b, cmap="Blues");

im2 = im.transpose(PIL.Image.ROTATE_90)



r, g, b = im2.split()



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))



axes[0].imshow(r, cmap="Reds");

axes[1].imshow(g, cmap="Greens");

axes[2].imshow(b, cmap="Blues");

im3 = im.convert("L")



plt.imshow(im3);
r, g, b = im.split()



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharex=True, sharey=True)



axes[0].hist(np.array(r).flatten(), bins=50);

axes[1].hist(np.array(g).flatten(), bins=50);

axes[2].hist(np.array(b).flatten(), bins=50);



axes[0].set_title("R");

axes[1].set_title("G");

axes[2].set_title("B");



for ax in axes:

    ax.set_ylabel("Frequency")

    ax.set_xlabel("Value")
r, g, b = im.split()

r = np.array(r)

g = np.array(g)

b = np.array(b)



r = (r - r.mean()) / r.std()

g = (g - g.mean()) / g.std()

b = (b - b.mean()) / b.std()



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharex=True, sharey=True)



axes[0].hist(r.flatten(), bins=30, color="k");

axes[1].hist(g.flatten(), bins=30, color="k");

axes[2].hist(b.flatten(), bins=30, color="k");



axes[0].set_title("Red band");

axes[1].set_title("Green band");

axes[2].set_title("Blue band");



axes[0].set_ylabel("Frequency")

axes[0].set_xlabel("Normalized value")

    

fig.subplots_adjust(wspace=0.05)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15), sharex=True, sharey=True)

for i, f in enumerate(glob.glob("/kaggle/input/ships-in-satellite-imagery/shipsnet/shipsnet/*.png")[0:25]):

    ax = axes[int(i/5)][int(i%5)]

    im = PIL.Image.open(f)

    ax.imshow(im)