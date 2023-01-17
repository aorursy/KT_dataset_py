# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

%matplotlib inline

init_notebook_mode(connected = True)

cf.go_offline()

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#This CSV file is separated by the 'delimiter ;'

df1 = pd.read_csv('/kaggle/input/eeg/extrait_wSleepPage01.csv', delimiter = ";")

df2 = pd.read_csv('/kaggle/input/eeg/spindles.csv', delimiter = ";")
df1
df2
print ('Start time of this record is', df1['Date'][0], df1['HH'][0],":",df1['MM'][0],":",df1['SS'][0])

print ('End time of this record is', df1['Date'].iloc[-1], df1['HH'].iloc[-1],":",df1['MM'].iloc[-1],":",df1['SS'].iloc[-1])

print ('Duration of measurement is about',df1['MM'].iloc[-1]-df1['MM'][0],'minutes')
#create a new column to match the time feature

df1['Time (s)'] = np.arange(0,248440*0.005,0.005)
# The useful features of df1 contain "EOG Left", "EEG C3-A1", "EEG O1-A1", "EEG C4-A1", "EEG O2-A1"

df1_new = df1[['Time (s)',"EOG Left", "EEG C3-A1", "EEG O1-A1", "EEG C4-A1", "EEG O2-A1"]]

df1_new
features = ["EOG Left", "EEG C3-A1", "EEG O1-A1", "EEG C4-A1", "EEG O2-A1"]

for x in features:

    df1_new[x] = [x.replace(',', '.') for x in df1_new[x]]



    

df1_new = df1_new.astype('float')
df1_new.info()
# plot these features in the same graph with stack plot

fig, axs = plt.subplots(5, sharex=True, sharey=True)

fig.set_size_inches(18, 24)

labels = ["EOG Left", "EEG C3-A1", "EEG O1-A1", 'EEG C4-A1', 'EEG O2-A1']

colors = ["r","g","b",'y',"k"]

fig.suptitle('Vertically stacked subplots of extrait_wSleepPage01', fontsize = 20)

# ---- loop over axes ----

for i,ax in enumerate(axs):

  axs[i].plot(df1_new['Time (s)'], df1_new[labels[i]],color=colors[i],label=labels[i])

  axs[i].legend(loc="upper right")



plt.xlabel('Time (s)', fontsize = 20)

plt.show()

#create a new column to match the time feature for df2

df2['Time (s)'] = np.arange(0,df2.count()[0]*0.005,0.005)
df2
# The useful features of df1 contain "EOG Left", "EEG C3-A1", "EEG O1-A1", "EEG C4-A1", "EEG O2-A1"

df2_new = df2[['Time (s)',"EOG Left[uV]", "EEG C3-A1[uV]", "EEG O1-A1[uV]", "EEG C4-A1[uV]", "EEG O2-A1[uV]"]]

df2_new
# replace the comma to dot

features = ["EOG Left[uV]", "EEG C3-A1[uV]", "EEG O1-A1[uV]", "EEG C4-A1[uV]", "EEG O2-A1[uV]"]

for x in features:

    df2_new[x] = [x.replace(',', '.') for x in df2_new[x]]



    

df2_new = df2_new.astype('float')
df2_new.info()
# plot these features in the same graph with stack plot

fig, axs = plt.subplots(5, sharex=True, sharey=True)

fig.set_size_inches(18, 24)

labels = ["EOG Left[uV]", "EEG C3-A1[uV]", "EEG O1-A1[uV]", 'EEG C4-A1[uV]', 'EEG O2-A1[uV]']

colors = ["r","g","b",'y',"k"]

fig.suptitle('Vertically stacked subplots of splendles', fontsize = 20)

# ---- loop over axes ----

for i,ax in enumerate(axs):

  axs[i].plot(df2_new['Time (s)'], df2_new[labels[i]],color=colors[i],label=labels[i])

  axs[i].legend(loc="upper right")

plt.xlabel('Time (s)', fontsize = 20)

plt.show()



from scipy import fft

df3 = df2_new.copy()

labels = ["EOG Left[uV]", "EEG C3-A1[uV]", "EEG O1-A1[uV]", 'EEG C4-A1[uV]', 'EEG O2-A1[uV]']

df3 = df3[labels].apply(fft)

df3['Time (s)'] = df2_new['Time (s)']

df3
# plot these FFT features in the same graph with stack plot

fig, axs = plt.subplots(5, sharex=True, sharey=True)

fig.set_size_inches(18, 24)

labels = ["EOG Left[uV]", "EEG C3-A1[uV]", "EEG O1-A1[uV]", 'EEG C4-A1[uV]', 'EEG O2-A1[uV]']

colors = ["r","g","b",'y',"k"]

fig.suptitle('Vertically stacked subplots of FFT', fontsize=20)

# ---- loop over axes ----

for i,ax in enumerate(axs):

  axs[i].plot(df3['Time (s)'], df3[labels[i]],color=colors[i],label=labels[i])

  axs[i].legend(loc="upper right")



plt.xlabel('Time (s)', fontsize = 20)

plt.show()
# add the feature of "frequency"

df3['frequency (Hz)'] = df3['Time (s)'].apply(lambda x: 1/x if x != 0 else 0 )

df3
# plot these FFT features in the same graph with stack plot

fig, axs = plt.subplots(5, sharex=True, sharey=True)

fig.set_size_inches(18, 24)

labels = ["EOG Left[uV]", "EEG C3-A1[uV]", "EEG O1-A1[uV]", 'EEG C4-A1[uV]', 'EEG O2-A1[uV]']

colors = ["r","g","b",'y',"k"]

fig.suptitle('Periodogram using FFT', fontsize=20)

# ---- loop over axes ----

for i,ax in enumerate(axs):

  axs[i].plot(df3['frequency (Hz)'][1:], df3[labels[i]][1:],color=colors[i],label=labels[i])

  axs[i].legend(loc="upper right")



plt.xlabel('Frequency (Hz)', fontsize = 20)

plt.show()