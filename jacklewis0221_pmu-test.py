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
!ls "/kaggle/input"
#Define quantity of data to be taken

numrows = 30000 # 0# #0



#Define variable to reduce resolution.

oursize = 1



#Function to define rows skipped if any.

def logic(index):

    if index <= (-0.112000000):

       return True

    elif index % oursize == 0:

       return False

    return True



import pandas as pd

import time



start = time.time()



PMU_Data = pd.read_csv('/kaggle/input/micro-pmu-october-1-dataset/_LBNL_a6_bus1_2015-10-01.csv',dtype='d', skiprows= lambda x: logic(x))





data_read = time.time()



print(data_read-start)



print(PMU_Data.head())
Mag_VA1 = np.array(PMU_Data['VL1'],dtype=np.float64)

Angle_VA1 = np.array(PMU_Data['AL1'],dtype=np.float64)

Angle_CA1 = np.array(PMU_Data['AC1'],dtype=np.float64)

Mag_VB1 = np.array(PMU_Data['VL2'],dtype=np.float64)

Mag_VC1 = np.array(PMU_Data['VL3'],dtype=np.float64)

Angle_VB1 = np.array(PMU_Data['AL2'],dtype=np.float64)

Angle_VC1 = np.array(PMU_Data['AL3'],dtype=np.float64)

Mag_CA1 = np.array(PMU_Data['IC1'],dtype=np.float64)

Mag_CB1 = np.array(PMU_Data['IC2'],dtype=np.float64)

Mag_CC1 = np.array(PMU_Data['IC3'],dtype=np.float64)

Angle_CB1 = np.array(PMU_Data['AC2'],dtype=np.float64)

Angle_CC1 = np.array(PMU_Data['AC3'],dtype=np.float64)

print(Angle_VA1)
from datetime import datetime

time = np.array(PMU_Data['Time']/1000000000,dtype=np.float64)

ourtime = np.empty([len(time)], dtype='datetime64[ms]')



for i in range(0, len(time)):

  ourtime[i] = datetime.utcfromtimestamp(int(time[i])).strftime('%Y-%m-%d %H:%M:%S.%f')



print(ourtime)

print(time[0])

print(datetime.utcfromtimestamp(int(time[0])).strftime('%Y-%m-%d %H:%M:%S.%f'))

print(datetime.utcfromtimestamp(int(time[len(time)-1])).strftime('%Y-%m-%d %H:%M:%S.%f'))





start = round((1100000)/oursize)

end = round((1180000)/oursize)
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 8))

plt.title('VA-Angle')

#plt.axvspan(prior_length, span, color=colour, alpha=0.5)

plt.plot(ourtime[start:end],Angle_VA1[start:end])

plt.grid(True)

plt.show()

plt.figure(figsize=(25, 8))

plt.title('CA-Angle')

#plt.axvspan(prior_length, span, color=colour, alpha=0.5)

plt.plot(ourtime[start:end],Angle_CA1[start:end])

plt.grid(True)

plt.show()

plt.figure(figsize=(25, 8))

plt.title('VA-Magnitude')

#plt.axvspan(prior_length, span, color=colour, alpha=0.5)

plt.plot(ourtime[start:end],Mag_VA1[start:end])

plt.grid(True)

plt.show()