# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plotting

 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Sample code to insert the RAW data into a dictionary

def read_raw_data(subject_number):

    

    if not(1 <= subject_number <= 16):

        raise ValueError("subject_number needs to be between 1 and 16")

    # We have two sessions for subject

    data = {}

    for session in range(2):

        session = session +1

        # read occipital data with SamplingInterval[ms]=2.000000

        occipital_data_1 =  pd.read_csv(f"/kaggle/input/eeg-looming/RAW_DATA/SUBJECT_{subject_number:02d}_SESSION_{session}.mul", skiprows=1,delim_whitespace=True,usecols=[7,15,23] ).values

        

        # read event file (has segmentations)

        evt_data_1 = pd.read_csv(f"/kaggle/input/eeg-looming/RAW_DATA/SUBJECT_{subject_number:02d}_SESSION_{session}.evt",delimiter='\t', skiprows=1, names=['TMU','CODE','Trigger','Comment',])

        evt_data_1['occipital_index'] = np.array(evt_data_1.TMU.values/2000, dtype=int)

        # Slice data holder

        looming_2,looming_3,looming_4 = [],[],[]



        min_l2,min_l3,min_l4 = np.inf,np.inf,np.inf

        iterator = evt_data_1.iterrows()

        for i,row in iterator:

            # Slice L2

            if "200s" in row.Comment:

                start = row.occipital_index

                end = start 

                while True:

                    _,row = next(iterator)

                    if 'stm-' in row.Comment:

                        end = row.occipital_index

                        break

                looming_2.append(occipital_data_1[start:end])

                min_l2 = np.min([min_l2,end-start])

            # Slice L3

            if "300s" in row.Comment:

                start = row.occipital_index

                end = start 

                while True:

                    _,row = next(iterator)

                    if 'stm-' in row.Comment:

                        end = row.occipital_index

                        break

                looming_3.append(occipital_data_1[start:end])

                min_l3 = np.min([min_l3,end-start])

            # Slice L4

            if "400s" in row.Comment:

                start = row.occipital_index

                end = start 

                while True:

                    _,row = next(iterator)

                    if 'stm-' in row.Comment:

                        end = row.occipital_index

                        break

                looming_4.append(occipital_data_1[start:end])

                min_l4 = np.min([min_l4,end-start])



        data[f'session_{session}_L2']=looming_2

        data[f'session_{session}_L3']=looming_3

        data[f'session_{session}_L4']=looming_4

    return data

# Sample code to read pre-cutted data

def load_looming(subject,session,looming,sample):

    filename = f"/kaggle/input/eeg-looming/LOOMING_DATA/SUBJECT_{subject:02d}/SESSION_{session}/L{looming}/{sample:02d}.csv"

    print(filename)

    sample = np.genfromtxt(filename, delimiter=',')

    return sample
# Plot a sample of cutted data

sample = load_looming(1,1,4,10)

time_ms = np.arange(sample.shape[0])*2

plt.plot(time_ms,sample[:,0],label='O1')

plt.plot(time_ms,sample[:,1],label='Oz')

plt.plot(time_ms,sample[:,2],label='O2')

plt.title('Cutted Sample 1-1-4-10')

plt.ylabel('Microvolts [uV]')

plt.xlabel('Miliseconds [ms]')

plt.legend()
# Plot a sample of raw data

raw_data  = read_raw_data(1)['session_1_L4'][9]

time_ms = np.arange(raw_data.shape[0])*2

plt.plot(time_ms,raw_data[:,0],label='O1')

plt.plot(time_ms,raw_data[:,1],label='Oz')

plt.plot(time_ms,raw_data[:,2],label='O2')

plt.title('Raw data Sample 1-1-4-10')

plt.ylabel('Microvolts [uV]')

plt.xlabel('Miliseconds [ms]')

plt.legend()