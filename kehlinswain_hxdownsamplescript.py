# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import scipy

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import scipy.signal as sg

def harvest_get_downsampled_signal(x, fs, target_fs):

    decimation_ratio = np.round(fs / target_fs)

    offset = np.ceil(140. / decimation_ratio) * decimation_ratio

    start_pad = x[0] * np.ones(int(offset), dtype=np.float32)

    end_pad = x.index[-1] * np.ones(int(offset), dtype=np.float32)

    x = np.concatenate((start_pad, x, end_pad), axis=0)



    if fs < target_fs:

        raise ValueError("CASE NOT HANDLED IN harvest_get_downsampled_signal")

    else:

        try:

            y0 = sg.decimate(x, int(decimation_ratio), 3, zero_phase=True)

        except:

            y0 = sg.decimate(x, int(decimation_ratio), 3)

        actual_fs = fs / decimation_ratio

        y = y0[int(offset / decimation_ratio):-int(offset / decimation_ratio)]

    y = y - np.mean(y)

    return y, actual_fs
import pandas as pd

calfRaise = pd.read_csv("/kaggle/input/calfraisenew/pressuretrial3calfraiseTest.csv")

calfRaise.columns
import pandas as pd

calfRaiseDF = pd.read_csv("../input/pressuretrial3calfraise.csv")

calfRaiseDF.head()

calfRaiseDF.index[0]

newDf = calfRaiseDF[1:]

newDf.head()
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
from scipy import signal 

med_gastro = calfRaise[' medGastro']

med_gastro.index[-1]

print(down_med_gastro[1])

down_med_gastro = down_med_gastro[0]

down_med_gastro
from sklearn import preprocessing

#saving for foot data

foot_position = calfRaise[' footPosMag']

df_foot = calfRaise[' footPosMag']

foot_position_normalize = (foot_position/foot_position.max())

print(foot_position.max())

foot_position = foot_position.values.reshape(-1,1)

#normalizing foot data

normalized_foot_position = preprocessing.normalize(foot_position)

normalized_foot_position[11]

foot_position_normalize.head()
# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Line chart showing how FIFA rankings evolved over time 

sns.lineplot(data=med_gastro, label="Med Gastro")

sns.lineplot(data=down_med_gastro, label="Down Sample")

sns.lineplot(data=resample_med_gastro, label="Resample")

sns.lineplot(data=foot_position_normalize, label="Foot Position")




#med gastro samples

down_med_gastro = harvest_get_downsampled_signal(med_gastro, 60, 50)

resample_med_gastro = signal.resample(med_gastro, 800)

resample_normalized = (resample_med_gastro/resample_med_gastro.max())



# Set the width and height of the figure

plt.figure(figsize=(16,6))

sns.lineplot(data=resample_normalized, label="Down Sample")

sns.lineplot(data=foot_position_normalize, label="Foot Position")



updateList = []

for i in range(397):

    updateList.append(0)

update_sample = np.append(resample_normalized,updateList)
# Set the width and height of the figure

plt.figure(figsize=(16,6))

sns.lineplot(data=update_sample, label="Down Sample")

sns.lineplot(data=foot_position_normalize, label="Foot Position")
# Set the width and height of the figure

plt.figure(figsize=(16,6))

#plt.ylim(0.8,1)

sns.scatterplot(x=update_sample, y=foot_position_normalize)

from sklearn.metrics import r2_score

r2_score(update_sample,foot_position_normalize)