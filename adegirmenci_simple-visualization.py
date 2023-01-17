import numpy as np # linear algebra
import pandas as pd # data processing

import h5py # to read HDF5 files
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

from os import walk, path

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

prepath = "../input" # file directory
Hfreq = 16384
Lfreq = 4096
def loadDataset( fileName ):
    "Reads data to memory"
    # https://losc.ligo.org/tutorial03/
    
    dataFile = h5py.File(path.join(prepath, fileName), 'r')
    gpsStart = dataFile['meta']['GPSstart'].value
    
    dqInfo = dataFile['quality']['simple']
    bitnameList = dqInfo['DQShortnames'].value
    nbits = len(bitnameList)
    
    strain = np.array(list(dataFile["/strain"].values())[0])
    
    for bit in range(nbits):
        print(bit, bitnameList[bit])
    print("-------------------")
    
    dataFile.close()

    return strain
files = []
for (dirpath, dirnames, filenames) in walk(prepath):
    files.extend(filenames)
print("Files in directory:")
print(*files, sep='\n')
print("-------------------")

df = []
for fname in files:
    df.append( loadDataset(fname) )
plt.figure(figsize = (14, 6))

Idx4Hz = []
Idx16Hz = []

for idx, fname in enumerate(files):
    time = np.linspace(0,32,len(df[idx]))
    if(fname[10] is "1"):
        plt.plot(time, df[idx], label = fname, markersize = 1)
        Idx16Hz.append(idx)
    else: # shift in y-axis to be able to see overlapping data
        plt.plot(time, df[idx]+1e-18, label = fname, markersize = 1)
        Idx4Hz.append(idx)

plt.legend()
plt.show()
# http://lexfridman.com/blogs/research/2015/09/18/fast-cross-correlation-and-time-series-synchronization-in-python/
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
 
def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x 
# shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift
shift4Hz = int(np.rint( compute_shift(df[Idx4Hz[0]],df[Idx4Hz[1]]) ))
shift16Hz = int(np.rint( compute_shift(df[Idx16Hz[0]],df[Idx16Hz[1]]) ))

shift4HzInSec = shift4Hz/Lfreq
shift16HzInSec = shift16Hz/Hfreq

print(" 4Hz: %f seconds of shift" % shift4HzInSec)
print("16Hz: %f seconds of shift" % shift16HzInSec)
dfAligned = np.zeros_like(df)
# crop to align signals
dfAligned[Idx4Hz[0]] = df[Idx4Hz[0]][0:-shift4Hz]
dfAligned[Idx4Hz[1]] = df[Idx4Hz[1]][shift4Hz:-1]

dfAligned[Idx16Hz[0]] = df[Idx16Hz[0]][0:-shift16Hz]
dfAligned[Idx16Hz[1]] = df[Idx16Hz[1]][shift16Hz:-1]
plt.figure(figsize = (14, 6))

for idx, fname in enumerate(files):
    if(fname[10] is "1"):
        time = np.linspace(0,32-shift16HzInSec,len(dfAligned[idx]))
        plt.plot(time, dfAligned[idx], label = fname, markersize = 1)
    else:
        time = np.linspace(0,32-shift4HzInSec,len(dfAligned[idx]))
        plt.plot(time, dfAligned[idx]+1e-18, label = fname, markersize = 1)

plt.legend()
plt.show()
# playground
dfAligned