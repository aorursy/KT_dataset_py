import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # charts
filename = r'/kaggle/input/emgstaticdynamic/badanie statyczne z obciazeniem.txt'
with open(filename, "r") as fin:
    rows = fin.readlines()[7:]
with open('static.txt', 'w') as fout:
    fout.writelines(rows)
    
static = pd.read_csv(r'/kaggle/working/static.txt', sep ='\t', header=None).drop(2,1)
static.columns = ['t[min]', 'U[mV]']

static['t[s]'] = static['t[min]']*60 #add column based on t[min] in seconds
static = static.drop('t[min]',1) #remove temporary column with time in [min] unit
static = static[['t[s]','U[mV]']] #reorder columns
static.head()
# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
static.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.05,ax=ax)
plt.show()
#1661,   6190
first = static[2200:6690]

# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
first.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.5,ax=ax)
plt.show()
def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'valid'))

windowSize = 1000;
staticLonger = static['U[mV]'].copy()
staticLonger = staticLonger.append(staticLonger[0:windowSize-1]) #expand series to add points for windowSize


staticRMSsignal = window_rms(staticLonger, windowSize)
staticRMS = static.copy()
staticRMS['U[mV]'] = staticRMSsignal
# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
staticRMS.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.5,ax=ax)
plt.show()
def split_above_threshold(signal, threshold):
    mask = np.concatenate(([False], signal > threshold, [False] ))
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    return idx


idx = split_above_threshold(staticRMSsignal, 0.008)
idx
windowSize = 30;
staticLonger = static['U[mV]'].copy()
staticLonger = staticLonger.append(staticLonger[0:windowSize-1]) #expand series to add points for windowSize


staticRMSsignal = window_rms(staticLonger, windowSize)
staticRMS = static.copy()
staticRMS['U[mV]'] = staticRMSsignal
# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
staticRMS.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.2,ax=ax)
plt.show()
signalOutT = [staticRMS['t[s]'][(idx[i]+800):(idx[i+1]+150)] for i in range(0,len(idx),2)]
signalOutU = [staticRMS['U[mV]'][(idx[i]+800):(idx[i+1]+150)] for i in range(0,len(idx),2)]
for i in range(len(signalOutT)):
  plt.plot(signalOutT[i],signalOutU[i])
for i in range(len(signalOutT)):
    plt.figure(figsize=(10,3))
    plt.plot(signalOutT[i],signalOutU[i],color="red")
    # Show/save figure as desired.
    plt.show()
    
for i in range(len(signalOutU)):
    print(np.mean(signalOutU[i],axis=0))
from scipy.signal import find_peaks
peaks, _ = find_peaks(staticRMSsignal, distance=15000)

# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
plt.plot(peaks, staticRMSsignal[peaks], "xr"); plt.plot(staticRMSsignal); plt.legend(['distance'])
plt.show()
print(staticRMSsignal[peaks])
filename = r'/kaggle/input/emgstaticdynamic/badanie dynamiczne z obciazeniem.txt'
with open(filename, "r") as fin:
    rows = fin.readlines()[7:]
with open('dynamic.txt', 'w') as fout:
    fout.writelines(rows)
    
dynamic = pd.read_csv(r'/kaggle/working/dynamic.txt', sep ='\t', header=None).drop(2,1)
dynamic.columns = ['t[min]', 'U[mV]'] #rename columns

dynamic['t[s]'] = dynamic['t[min]']*60 #add column based on t[min] in seconds
dynamic = dynamic.drop('t[min]',1) #remove temporary column with time in [min] unit
dynamic = dynamic[['t[s]','U[mV]']] #reorder columns
dynamic.head()
# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
dynamic.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.25,ax=ax)
plt.show()
windowSize = 1000;
dynamicLonger = dynamic['U[mV]'].copy()
dynamicLonger = dynamicLonger.append(dynamicLonger[0:windowSize-1]) #expand series to add points for windowSize


dynamicRMSsignal = window_rms(dynamicLonger, windowSize)
dynamicRMS = dynamic.copy()
dynamicRMS['U[mV]'] = dynamicRMSsignal


# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
dynamicRMS.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.2,ax=ax)
plt.show()

idx = split_above_threshold(dynamicRMSsignal, 0.008)

windowSize = 30;
dynamicLonger = dynamic['U[mV]'].copy()
dynamicLonger = dynamicLonger.append(dynamicLonger[0:windowSize-1]) #expand series to add points for windowSize


dynamicRMSsignal = window_rms(dynamicLonger, windowSize)
dynamicRMS = dynamic.copy()
dynamicRMS['U[mV]'] = dynamicRMSsignal

# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
dynamicRMS.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.2,ax=ax)
plt.show()

signalOutT = [dynamicRMS['t[s]'][(idx[i]+800):(idx[i+1]+150)] for i in range(0,len(idx),2)]
signalOutU = [dynamicRMS['U[mV]'][(idx[i]+800):(idx[i+1]+150)] for i in range(0,len(idx),2)]


for i in range(len(signalOutT)):
  plt.plot(signalOutT[i],signalOutU[i])

for i in range(len(signalOutT)):
    plt.figure(figsize=(10,3))
    plt.plot(signalOutT[i],signalOutU[i],color="red")
    # Show/save figure as desired.
    plt.show()
    


from scipy.signal import find_peaks
peaks, _ = find_peaks(dynamicRMS['U[mV]'], distance=15000)

# a scatter plot
f, ax = plt.subplots(1, 1, figsize = (35, 10))
plt.plot((peaks/1000), dynamicRMS['U[mV]'][peaks], "xb"); dynamicRMS.plot(kind='line',x='t[s]',y='U[mV]',color='red',linewidth=.2,ax=ax); plt.legend(['Peak-to-peak'])
plt.show()

print(dynamicRMSsignal[peaks])