# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/signal-processing'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Lets insert the necessary libraries to work with 
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import matplotlib as mpl

from scipy.stats import norm
import seaborn as sns
from scipy import stats
file_1 = pd.read_table('../input/signal-processing/fmserp1901.txt', sep="\t", header=None)
file_1.head()
file_1 = pd.read_table('../input/signal-processing/fmserp1901.txt', sep="\t", header=None)
file_2 = pd.read_table('../input/signal-processing/fmserp1902.txt', sep="\t", header=None)
file_3 = pd.read_table('../input/signal-processing/fmserp1903.txt', sep="\t", header=None)
result =pd.concat([file_1, file_2, file_3], ignore_index=True)
# Replacing the comma values 

vol = result.iloc[:,0] =result.iloc[:,0].str.replace(',', '.').astype(float)
vol=abs(vol)
pa = result.iloc[:,1] = result.iloc[:,1].str.replace(',', '.').astype(float)
pc = result.iloc[:,2] = result.iloc[:,2].str.replace(',', '.').astype(float)
i  = result.iloc[:,3] = result.iloc[:,3].str.replace(',', '.').astype(float)
result.shape
result.head()
result.describe()
result.columns = ['voltage', 'pres_inlet', 'pres_outlet', 'elec']
# result.plot(subplots = True )
result.plot(subplots = True , figsize=(12,8));
t= len(result)
test_list = [0 + (x * 0.00048828125) for x in range(0, t)]
x = test_list[0:t]
fig = plt.figure(figsize =  (12,5))
plt.plot(x,vol, c='black', label = 'Voltage' )
plt.xticks(rotation=90, fontsize = 16, fontname = 'Arial')
plt.yticks(rotation=90,fontsize = 16, fontname = 'Arial')
plt.xlabel('Time / [s]', fontsize = 16)
plt.ylabel('Voltage / [Volt]', fontsize = 16)
plt.legend(loc='upper right', fontsize = 16)
plt.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
vol.skew()

vol.kurt()
# Normal distribution for the voltage 

sns.distplot(vol, fit=norm, color="r")

# Normal distribution for the voltage 

sns.distplot(pa, fit=norm, color="g")


# Normal distribution for the voltage 

sns.distplot(pc, fit=norm, color="k")
import seaborn as sns
corrmat = result.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=0.7, square=True, cmap="cubehelix", annot=True);
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

import seaborn as sns
corrmat = result.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=0.7, square=True, cmap="plasma", annot=True);
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

vol_80 = signal.detrend(vol, axis = -1,type = 'linear', bp =  np.arange(0,len(vol),80*2048))
vol_5 = signal.detrend(vol, axis = -1,type = 'linear', bp =  np.arange(0,len(vol),5*2048))

fig = plt.figure(figsize =  (12,5))

plt.plot(x,vol, c='black', label = 'Voltage actual' )
plt.plot(x,vol_80, c='blue', label = 'Voltage (80s window)' )
plt.plot(x,vol_5, c='red', label = 'Voltage (5s window)' )


plt.xticks(rotation=90, fontsize = 16, fontname = 'Arial')
plt.yticks(rotation=90,fontsize = 16, fontname = 'Arial')
plt.xlabel('Time / [s]', fontsize = 16)
plt.ylabel('Voltage / [Volt]', fontsize = 16)
plt.legend(loc='upper right', fontsize = 16)
plt.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
pa_80 = signal.detrend(pa, axis = -1,type = 'linear', bp =  np.arange(0,len(pa),80*2048))
pa_5 = signal.detrend(pa, axis = -1,type = 'linear', bp =  np.arange(0,len(pa),5*2048))

fig = plt.figure(figsize =  (12,6))

plt.plot(x,pa, c='black', label = 'Pres_anode actual' )
plt.plot(x,pa_80, c='blue', label = 'Pres_anode (80s window)' )
plt.plot(x,pa_5, c='red', label = 'Pres_anode (5s window)' )


plt.xticks(rotation=90, fontsize = 16, fontname = 'Arial')
plt.yticks(rotation=90,fontsize = 16, fontname = 'Arial')
plt.xlabel('Time / [s]', fontsize = 16)
plt.ylabel('Voltage / [Volt]', fontsize = 16)
plt.legend(loc='upper right', fontsize = 16)
plt.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
pc_80 = signal.detrend(pa, axis = -1,type = 'linear', bp =  np.arange(0,len(pc),80*2048))
pc_5  = signal.detrend(pa, axis = -1,type = 'linear', bp =  np.arange(0,len(pc),5*2048))

fig = plt.figure(figsize =  (12,6))

plt.plot(x,pc, c='black', label = 'Pres_cathode actual' )
plt.plot(x,pc_80, c='blue', label = 'Pres_cathode (80s window)' )
plt.plot(x,pc_5, c='red', label = 'Pres_cathode (5s window)' )


plt.xticks(rotation=90, fontsize = 16, fontname = 'Arial')
plt.yticks(rotation=90,fontsize = 16, fontname = 'Arial')
plt.xlabel('Time / [s]', fontsize = 16)
plt.ylabel('Voltage / [Volt]', fontsize = 16)
plt.legend(loc='upper right', fontsize = 16)
plt.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
sns.distplot(vol_5, fit=norm, color="r")
sns.distplot(pa_5, fit=norm, color="k")
sns.distplot(pc_5, fit=norm, color="g")
parts = 200
two_split = np.array_split(vol_5, parts)

b3 = []
for array in two_split:
    a =np.std(array)
    b3.append(a)
x1 = np.linspace(0, 1800, parts)


mpl.rcParams['font.size']=16
plt.rc('font', size=16)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 4


fig = plt.figure(figsize =  (12, 6))


plt.plot(x1,b3, c='red', label = 'Voltage')
plt.xlabel('Time / s',  fontname = 'Arial', weight = 'bold')
plt.ylabel('STD / Bar',  fontname = 'Arial', weight = 'bold')
plt.legend(loc='upper right', fontsize = 16)

plt.xticks( fontname = 'Arial', weight = 'bold')


plt.yticks( fontname = 'Arial', weight = 'bold')
plt.ticklabel_format(axis='y', scilimits=(0,0))
parts = 200
two_split = np.array_split(pa_5, parts)

b3 = []
for array in two_split:
    a =np.std(array)
    b3.append(a)
x1 = np.linspace(0, 1800, parts)


mpl.rcParams['font.size']=16
plt.rc('font', size=16)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 4


fig = plt.figure(figsize =  (12, 6))


plt.plot(x1,b3, c='red', label = 'Pressure anode')
plt.xlabel('Time / s',  fontname = 'Arial', weight = 'bold')
plt.ylabel('STD / Bar',  fontname = 'Arial', weight = 'bold')
plt.legend(loc='upper right', fontsize = 16)

plt.xticks( fontname = 'Arial', weight = 'bold')


plt.yticks( fontname = 'Arial', weight = 'bold')
plt.ticklabel_format(axis='y', scilimits=(0,0))
parts = 200
two_split = np.array_split(pc_5, parts)

b3 = []
for array in two_split:
    a =np.std(array)
    b3.append(a)
x1 = np.linspace(0, 1800, parts)


mpl.rcParams['font.size']=16
plt.rc('font', size=16)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 4


fig = plt.figure(figsize =  (12, 6))


plt.plot(x1,b3, c='red', label = 'Pressure cathode')
plt.xlabel('Time / s',  fontname = 'Arial', weight = 'bold')
plt.ylabel('STD / Bar',  fontname = 'Arial', weight = 'bold')
plt.legend(loc='upper right', fontsize = 16)

plt.xticks( fontname = 'Arial', weight = 'bold')


plt.yticks( fontname = 'Arial', weight = 'bold')
plt.ticklabel_format(axis='y', scilimits=(0,0))