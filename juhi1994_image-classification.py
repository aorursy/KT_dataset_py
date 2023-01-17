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
import cv2

sample_path='/kaggle/input/intel-image-classification/seg_test/seg_test/sea/21192.jpg'

image_data=cv2.imread(sample_path)

type(image_data)

print(image_data.shape)
import matplotlib.pyplot as plt

plt.imshow(image_data)
image_reshaped=image_data.reshape(1,150*150*3)

image_reshaped
sample_path.split('/')[-1]