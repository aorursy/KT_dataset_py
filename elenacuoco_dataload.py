# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#########
# you need to download such code https://losc.ligo.org/s/sample_code/readligo.py
# to read the data
##########
import matplotlib.pyplot as plt
# LIGO-specific readligo.py 
import numpy as np
import os
import fnmatch 
    
def read_hdf5(filename, readstrain=True):
    """
    Helper function to read HDF5 files
    """
    import h5py
    dataFile = h5py.File(filename, 'r')

    #-- Read the strain
    if readstrain:
        strain = dataFile['strain']['Strain'][...]
    else:
        strain = 0

    ts = dataFile['strain']['Strain'].attrs['Xspacing']
    
    #-- Read the DQ information
    dqInfo = dataFile['quality']['simple']
    qmask = dqInfo['DQmask'][...]
    shortnameArray = dqInfo['DQShortnames'].value
    shortnameList  = list(shortnameArray)
    
    # -- Read the INJ information
    injInfo = dataFile['quality/injections']
    injmask = injInfo['Injmask'][...]
    injnameArray = injInfo['InjShortnames'].value
    injnameList  = list(injnameArray)
    
    #-- Read the meta data
    meta = dataFile['meta']
    gpsStart = meta['GPSstart'].value    
    
    dataFile.close()
    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList

def loaddata(filename, ifo=None, tvec=True, readstrain=True):
    """
    The input filename should be a LOSC .hdf5 file 
     
    The detector should be H1, H2, or L1.

    The return value is: 
    STRAIN, TIME, CHANNEL_DICT

    STRAIN is a vector of strain values
    TIME is a vector of time values to match the STRAIN vector
         unless the flag tvec=False.  In that case, TIME is a
         dictionary of meta values.
    CHANNEL_DICT is a dictionary of data quality channels    
    """

    # -- Check for zero length file
    if os.stat(filename).st_size == 0:
        return None, None, None

    file_ext = os.path.splitext(filename)[1]    
     
    strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = read_hdf5(filename, readstrain)
        
    #-- Create the time vector
    gpsEnd = gpsStart + len(qmask)
    if tvec:
        time = np.arange(gpsStart, gpsEnd, ts)
    else:
        meta = {}
        meta['start'] = gpsStart
        meta['stop']  = gpsEnd
        meta['dt']    = ts

    #-- Create 1 Hz DQ channel for each DQ and INJ channel
    channel_dict = {}  #-- 1 Hz, mask
    slice_dict   = {}  #-- sampling freq. of stain, a list of slices
    final_one_hz = np.zeros(qmask.shape, dtype='int32')
    for flag in shortnameList:
        bit = shortnameList.index(flag)
        # Special check for python 3
        if isinstance(flag, bytes): flag = flag.decode("utf-8") 
        
        channel_dict[flag] = (qmask >> bit) & 1

    for flag in injnameList:
        bit = injnameList.index(flag)
        # Special check for python 3
        if isinstance(flag, bytes): flag = flag.decode("utf-8") 
        
        channel_dict[flag] = (injmask >> bit) & 1
       
    #-- Calculate the DEFAULT channel
    try:
        channel_dict['DEFAULT'] = ( channel_dict['DATA'] )
    except:
        print("Warning: Failed to calculate DEFAULT data quality channel")

    if tvec:
        return strain, time, channel_dict
    else:
        return strain, meta, channel_dict
#----------------------------------------------------------------
# Load LIGO data from a single file
#----------------------------------------------------------------
# First from H1
fn_H1 = '../input/H-H1_LOSC_4_V1-1126259446-32.hdf5'
strain_H1, time_H1, chan_dict_H1 = loaddata(fn_H1, 'H1')
# and then from L1
fn_L1 = '../input/L-L1_LOSC_4_V1-1126259446-32.hdf5'
strain_L1, time_L1, chan_dict_L1 = loaddata(fn_L1, 'L1')

# sampling rate:
fs = 4096
# both H1 and L1 will have the same time vector, so:
time = time_H1
# the time sample interval (uniformly sampled!)
dt = time[1] - time[0]

# plot +- 5 seconds around the event:
tevent = 1126259462.422         # Mon Sep 14 09:50:45 GMT 2015 
deltat = 5.                     # seconds around the event
# index into the strain time series for this time interval:
indxt = np.where((time_H1 >= tevent-deltat) & (time_H1 < tevent+deltat))

plt.figure()
plt.plot(time_H1[indxt]-tevent,strain_H1[indxt],'r',label='H1 strain')
plt.plot(time_L1[indxt]-tevent,strain_L1[indxt],'g',label='L1 strain')
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('strain')
plt.legend(loc='lower right')
plt.title('Advanced LIGO strain data near GW150914')
plt.savefig('GW150914_strain.png')