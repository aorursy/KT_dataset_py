%matplotlib inline

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
from PIL import Image

import math

import matplotlib.pyplot as plt

import random



def aspect_ratio_histogram(folder_path, bins=None):

    aspect_ratios = []

    images_path = list(folder_path.glob("*.png"))

    random.shuffle(images_path)

    print(len(images_path))

    for image_path in images_path:

        image = np.asarray(Image.open(str(image_path)))

        assert(len(image.shape)==3) # (W, H, C)

        # aspect_ratios.append(image.shape[0] / image.shape[1])

        aspect_ratios.append(math.log10(image.shape[0] / image.shape[1]))

    aspect_ratios = np.asarray(aspect_ratios)

    n, bins, patches = plt.hist(aspect_ratios, bins=bins, log=True, alpha=0.5)

    return bins



from pathlib import Path

train_path = Path("../input/train/")

test_path = Path("../input/test/")

bins = aspect_ratio_histogram(train_path, bins=100)

aspect_ratio_histogram(test_path, bins=bins)

plt.savefig('aspect_ratio_histogram.png')