# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from scipy.stats import zscore

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
air = pd.read_csv('../input/sofia-air-quality-dataset/2017-12_sds011sof.csv')
air.head()
plt.plot(air.P1)
plt.plot(air.P2)
bme = pd.read_csv('../input/sofia-air-quality-dataset/2017-12_bme280sof.csv')
bme.head()
plt.plot(bme.temperature)
plt.plot(bme.humidity)
plt.plot(bme.pressure)