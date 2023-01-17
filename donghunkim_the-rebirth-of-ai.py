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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_sample_images



dataset = load_sample_images()   

img_rgb = dataset.images[1]

img_rgb.shape
plt.figure(figsize=(25, 20))



plt.subplot(141)

plt.imshow(img_rgb[0:427, 0:640, :])

plt.axis("off")

plt.title("RGB image")



plt.subplot(142)

plt.imshow(img_rgb[0:427, 0:640, 0], cmap=plt.cm.bone)

plt.axis("off")

plt.title("R")



plt.subplot(143)

plt.imshow(img_rgb[0:427, 0:640, 1], cmap=plt.cm.bone)

plt.axis("off")

plt.title("G")



plt.subplot(144)

plt.imshow(img_rgb[0:427, 0:640, 2], cmap=plt.cm.bone)

plt.axis("off")

plt.title("B ")



plt.show()
img_rgb[0:427, 0:640, 1]