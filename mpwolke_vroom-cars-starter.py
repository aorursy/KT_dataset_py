# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-car-connection-picture-dataset/BMW_6-Series_2019_81_18_310_30_6_74_54_197_20_RWD_5_4_4dr_QmJ.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-car-connection-picture-dataset/Cadillac_ATS_2016_44_18_270_20_4_72_55_183_nan_AWD_4_2_2dr_NPH.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-car-connection-picture-dataset/Porsche_Taycan_2020_103_19_520_nan_nan_77_54_195_nan_AWD_4_4_4dr_Scl.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)