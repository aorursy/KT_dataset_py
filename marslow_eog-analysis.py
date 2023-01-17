#pip install bioread
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bioread # utilities for reading the files produced by BIOPAC's AcqKnowledge software
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # often used function 
from scipy.ndimage.filters import gaussian_filter1d #for performing deritave of Gaussian wavelet convolution

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
fig, axs = plt.subplots(5, 1)
fig.set_size_inches(25, 10)
#fig.suptitle('Sygnały EOG (.05-35 Hz) (mV)')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(top=3.5)

for ax in axs:
    ax.set_xlabel('czas [s]', size=20)
    ax.set_ylabel('napięcie [mV]', size=20)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data = bioread.read_file(os.path.join(dirname, filename))
        testname = os.path.splitext(filename)[0]
        
        if testname.startswith('tekst wolny'):
            plt.sca(axs[0])
        elif testname.startswith('tekst szybki'):
            plt.sca(axs[1])
        elif testname.startswith('tekst b_szybki'):
            plt.sca(axs[2])
        elif testname.startswith('sakady'):
            plt.sca(axs[3])
        else:
            plt.sca(axs[4])
            
        for chan in data.channels:
            plt.plot(chan.time_index, chan.data, label='{}'.format(testname))
        
        plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=3,fontsize=20,borderaxespad=0, frameon=False)
filename = r'/kaggle/input/eog-data/nocna.acq'
testname = os.path.splitext(filename)[0]
testname = os.path.basename(testname)

# This file is included in dataset
data = bioread.read_file(filename)
f, ax = plt.subplots(1, 1, figsize = (35, 10))
plt.xlabel('czas [s]', size=20)
plt.ylabel('napięcie [mV]',size=20)
           
for chan in data.channels:
    plt.plot(chan.time_index, chan.data, label='{}'.format(testname))
    peaks, properties = find_peaks(chan.data, height=0.1, distance=(30*500))
    plt.plot((peaks/500), chan.data[peaks], "xr",markersize=20)
    plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)
    
nocna = properties["peak_heights"]
filename = r'/kaggle/input/eog-data/dzienna.acq'
testname = os.path.splitext(filename)[0]
testname = os.path.basename(testname)

# This file is included in dataset
data = bioread.read_file(filename)
f, ax = plt.subplots(1, 1, figsize = (35, 10))
plt.xlabel('czas [s]', size=20)
plt.ylabel('napięcie [mV]',size=20)
for chan in data.channels:
    plt.plot(chan.time_index, chan.data, label='{}'.format(testname))
    peaks, properties = find_peaks(chan.data, height=0.1, distance=(30*500))
    plt.plot((peaks/500), chan.data[peaks], "xr",markersize=20)
    plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)

    
dzienna = properties["peak_heights"]
nocna_sort = np.sort(nocna)
dzienna_sort = np.sort(dzienna)
dzienna_sort = dzienna_sort[::-1] #reverse order (descending)

data = np.array([nocna_sort[:-1], dzienna_sort]) #remove last element after sorting from nocna array to keep columns the same size

dataset = pd.DataFrame({'nocna [mV]': data[0], 'dzienna [mV]': data[1]})
dataset.index += 1

arden = dataset['dzienna [mV]'][1]/dataset['nocna [mV]'][1]

print('Arden coefficient: ',arden)

pd.set_option('max_rows', None)
dataset
filename = r'/kaggle/input/eog-data/sakady_pacjent_A.acq'
testname = os.path.splitext(filename)[0]
testname = os.path.basename(testname)

start = 8 #cut noise signal from the beginning
end = 50 #and end of the eog signal
res = 500 #resolution = 500 samples/s

# This file is included in dataset
data = bioread.read_file(filename)
f, ax = plt.subplots(1, 1, figsize = (35, 10))
plt.xlabel('czas [s]', size=20)
plt.ylabel('napięcie [mV]',size=20)
for chan in data.channels:
    x = chan.time_index[start*res:end*res]
    y = chan.data[start*res:end*res]
    
    plt.plot(x, y, label='{}'.format(testname))
    peaks, properties = find_peaks(y, prominence=0.1, distance=(3*res), width=(0.2*res), wlen=1000, rel_height=0.4)
    
    negative_peaks, negative_properties = find_peaks(-y, prominence=0.1, distance=(3*res), width=(0.2*res), wlen=2500, rel_height=0.3)
    
    #plot positive peaks
    plt.plot(x[peaks], y[peaks], "xr",markersize=20)
    plt.vlines(x=x[peaks], ymin=y[peaks] - properties["prominences"],ymax = y[peaks], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"]/res+start, xmax=properties["right_ips"]/res+start, color = "C1")
    
    #plot negative peaks
    plt.plot(x[negative_peaks], y[negative_peaks], "xr",markersize=20)
    plt.vlines(x=x[negative_peaks], ymin=0, ymax = y[negative_peaks], color = "C1")
    plt.hlines(y=-negative_properties["width_heights"], xmin=negative_properties["left_ips"]/res+start, xmax=negative_properties["right_ips"]/res+start, color = "C1")
    
    plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)
    
durationA_left = properties["widths"]/res*1000 # duartion of saccades in ms
durationA_right = negative_properties["widths"]/res*1000

print(durationA_left)
print(durationA_right)

filename = r'/kaggle/input/eog-data/sakady_pacjent_B.acq'
testname = os.path.splitext(filename)[0]
testname = os.path.basename(testname)

start = 67 #cut noise signal from the beginning (10 for first serie)
end = 110 #and end of the eog signal (60 for first serie)
res = 500 #resolution = 500 samples/s

# This file is included in dataset
data = bioread.read_file(filename)
f, ax = plt.subplots(1, 1, figsize = (35, 10))
plt.xlabel('czas [s]', size=20)
plt.ylabel('napięcie [mV]',size=20)
for chan in data.channels:
    x = chan.time_index[start*res:end*res]
    y = chan.data[start*res:end*res]
    
    plt.plot(x, y, label='{}'.format(testname))
    peaks, properties = find_peaks(y, prominence=0.1, distance=(3*res), width=(0.1*res), wlen=1500, rel_height=0.55)
    
    negative_peaks, negative_properties = find_peaks(-y, prominence=0.1, distance=(3*res), width=(0.2*res), wlen=2500, rel_height=0.37)
    
    #plot positive peaks
    plt.plot(x[peaks], y[peaks], "xr",markersize=20)
    plt.vlines(x=x[peaks], ymin=y[peaks] - properties["prominences"],ymax = y[peaks], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"]/res+start, xmax=properties["right_ips"]/res+start, color = "C1")
    
    #plot negative peaks
    plt.plot(x[negative_peaks], y[negative_peaks], "xr",markersize=20)
    plt.vlines(x=x[negative_peaks], ymin=0, ymax = y[negative_peaks], color = "C1")
    plt.hlines(y=-negative_properties["width_heights"], xmin=negative_properties["left_ips"]/res+start, xmax=negative_properties["right_ips"]/res+start, color = "C1")
    
    plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)
    

durationB_left = properties["widths"]/res*1000 # duartion of saccades
durationB_right = negative_properties["widths"]/res*1000

print(durationB_left)
print(durationB_right)

#data = np.array([Sakada_lewa, Sakada_prawa]) #remove last element after sorting from nocna array to keep columns the same size
angle = [15,15,15,30,30,30,45,45,45]

angular_velocityLA = angle/durationA_left
angular_velocityRA = angle/durationA_right

angular_velocityLB = angle/durationB_left
angular_velocityRB = angle/durationB_right

datasetA = pd.DataFrame({'kąt [°]': angle,'Sakada_lewa [ms]': durationA_left, 'Sakada_prawa [ms]': durationA_right, 'prędkośc_lewa [°/ms]': angular_velocityLA, 'prędkośc_prawa [°/ms]': angular_velocityRA})
datasetA['Sakada_lewa [ms]'] = round(datasetA['Sakada_lewa [ms]']).astype(int)
datasetA['Sakada_prawa [ms]'] = round(datasetA['Sakada_prawa [ms]']).astype(int)

datasetB = pd.DataFrame({'kąt [°]': angle,'Sakada_lewa [ms]': durationB_left, 'Sakada_prawa [ms]': durationB_right,'prędkośc_lewa [°/ms]': angular_velocityLB, 'prędkośc_prawa [°/ms]': angular_velocityRB})
datasetB['Sakada_lewa [ms]'] = round(datasetB['Sakada_lewa [ms]']).astype(int)
datasetB['Sakada_prawa [ms]'] = round(datasetB['Sakada_prawa [ms]']).astype(int)

datasetA.index += 1
datasetB.index += 1
datasetB['Sakada_prawa [ms]'] = round(datasetB['Sakada_prawa [ms]']).astype(int)
pd.set_option('max_rows', None,'precision',3)

datasetA
datasetB

"""
Thomas Kahn
thomas.b.kahn@gmail.com
https://github.com/thomasbkahn/step-detect/blob/master/step_detect.py
"""

def find_steps(array, threshold):
    """
    Finds local maxima by segmenting array based on positions at which
    the threshold value is crossed. Note that this thresholding is 
    applied after the absolute value of the array is taken. Thus,
    the distinction between upward and downward steps is lost. However,
    get_step_sizes can be used to determine directionality after the
    fact.

    Parameters
    ----------
    array : numpy array
        1 dimensional array that represents time series of data points
    threshold : int / float
        Threshold value that defines a step


    Returns
    -------
    steps : list
        List of indices of the detected steps

    """
    steps        = []
    array        = np.abs(array)
    above_points = np.where(array > threshold, 1, 0)
    ap_dif       = np.diff(above_points)
    cross_ups    = np.where(ap_dif == 1)[0]
    cross_dns    = np.where(ap_dif == -1)[0]
    for upi, dni in zip(cross_ups,cross_dns):
        steps.append(np.argmax(array[upi:dni]) + upi)
    return steps
plt.rcParams.update({'figure.max_open_warning': 0}) #supress warnings about too many plots
res = 500 #resolution = 500 samples/s

# create dictionary of DataFrames
d = {}

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:        
        testname = os.path.splitext(filename)[0]
                
        if testname.startswith('tekst'):  
            data = bioread.read_file(os.path.join(dirname, filename))
            
            
            #print('Analysed signal: ',filename)
            #print()
            
            f, ax = plt.subplots(1, 1, figsize = (35, 5))
        
            for chan in data.channels:
                start=0
                end=len(chan.time_index)
                x = chan.time_index[int(start*res):end*res]
                y = chan.data[int(start*res):end*res]

                plt.plot(x, y, label='{}'.format(testname))
                peaks, properties = find_peaks(y, prominence=0.1, distance=(0.3*res), width=(0.4*res), wlen=2500, rel_height=0.8)

                #plot positive peaks
                plt.plot(x[peaks], y[peaks], "xr",markersize=20)
                plt.vlines(x=x[peaks], ymin=y[peaks] - properties["prominences"],ymax = y[peaks], color = "C1")
                plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"]/res+start, xmax=properties["right_ips"]/res+start, color = "C1")

                plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)


            #analyze saccades in each textline
            #extract signal according to lines

            n = len(properties["right_ips"]) # number of lines
            #print("Number of lines in text: ", n)
            #print()
            
            time_read = []
            fix_n = []
            avg_time = []

            for i in range(n):
                b = int(properties["left_ips"][i])
                e = int(properties["right_ips"][i])
                xx=x[b:e]
                yy=y[b:e]
                
                time_read.append((e-b)/res)

                # Deritave of Gaussian wavelet convolution can be done with a SciPy function 

                dg1 = gaussian_filter1d(yy, 8, order=1)
                dg1 /= np.abs(dg1).max() # normalizing here to facillitate comparison despite vastly varying magnitudes
                indxs = find_steps(np.abs(dg1), 0.1)

                fig, axs = plt.subplots(2, 1)
                fig.set_size_inches(10, 4)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.subplots_adjust(top=3.5)

                plt.sca(axs[0])
                plt.plot(xx, yy, label='line {}'.format(i+1))
                plt.plot(xx[indxs], yy[indxs], "xr",markersize=20)
                plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)

                plt.sca(axs[1])
                plt.plot(xx, dg1, label='line {} step detection'.format(i+1))
                plt.plot(xx[indxs], dg1[indxs], "xr",markersize=20)
                plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0),ncol=1,fontsize=20,borderaxespad=0, frameon=False)

                #print('Indexes of steps positions: ', indxs)
                
                fix_num = len(indxs)-1 #Number of fixations in line
                #print('Number of fixations in line {}: '.format(i+1),fix_num)
                
                fix_n.append(fix_num)

                #duration = indxs[1:] - indxs[:-1]
                duration = [(x1 - x2)*2 for (x1, x2) in zip(indxs[1:], indxs[:-1])] #(x1-x2) divided by 2 because sample rate is 500/s and we multiply by 1000 to convert to ms 
                mean_time = np.mean(duration)
                #print('Duration of fixations in line {} in ms: '.format(i+1),duration)
                #print('Average time of fixation in line {} in ms: '.format(i+1),mean_time)
                avg_time.append(mean_time)
                
                #print()
                #print()
                #print()
                
                
            dataf = pd.DataFrame({'Czas czytania lini tekstu [s] ': time_read, 'Liczba fiksacji': fix_n, 'Średni czas fiksacji [ms]': avg_time})
            dataf.index +=1
            dataf.index.name = 'Nr lini'
            
            # Get names of indexes for which column Age has value 30
            indexNames = dataf[ dataf['Liczba fiksacji'] > 10 ].index
            # Delete these row indexes from dataFrame
            dataf.drop(indexNames , inplace=True) 
            
            dataf.loc['średnia'] = dataf.mean()
                        
            dataf['Liczba fiksacji'] = round(dataf['Liczba fiksacji']).astype(int)
            dataf['Średni czas fiksacji [ms]'] = round(dataf['Średni czas fiksacji [ms]']).astype(int)
            


            d[testname] = dataf
            
pd.set_option('max_rows', None,'precision',2)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:        
        testname = os.path.splitext(filename)[0]
        if testname.startswith('tekst'):
            print(testname)
            display(d.get(testname))