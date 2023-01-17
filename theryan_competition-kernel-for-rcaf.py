import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models, layers, optimizers, utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.rcParams['figure.figsize'] = [10, 5]
trn = pd.read_csv('../input/train.csv')
trn.head()
trn.describe(percentiles=[0.002,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.998])
# checkout histograms to check distributions
plt.rcParams['figure.figsize'] = [10, 5]
trn.hist(['eeg_fp1','eeg_f7','eeg_f8','eeg_t4','eeg_t6','eeg_t5','eeg_t3','eeg_fp2','eeg_o1','eeg_p3','eeg_pz','eeg_f3','eeg_fz','eeg_f4','eeg_c4','eeg_p4','eeg_poz','eeg_c3','eeg_cz','eeg_o2','ecg','r','gsr'])
plt.show()
# plot some segment

crw = trn[trn['crew'] == 1] # plot data of just one crew
eegs = ['eeg_fp1','eeg_f7','eeg_f8','eeg_t4','eeg_t6','eeg_t5','eeg_t3','eeg_fp2','eeg_o1','eeg_p3','eeg_pz','eeg_f3','eeg_fz','eeg_f4','eeg_c4','eeg_p4','eeg_poz','eeg_c3','eeg_cz','eeg_o2']

for eeg in eegs:
    plt.plot(crw.iloc[208000:212000,:][eeg])
plt.show()
# large outliers and plenty of noise in eeg data, so first smooth data, then cap eeg data at some percentile
for eeg in ['eeg_fp1','eeg_f7','eeg_f8','eeg_t4','eeg_t6','eeg_t5',
            'eeg_t3','eeg_fp2','eeg_o1','eeg_p3','eeg_pz','eeg_f3',
            'eeg_fz','eeg_f4','eeg_c4','eeg_p4','eeg_poz','eeg_c3',
            'eeg_cz','eeg_o2']:
    # two rounds of smoothing
    trn[eeg] = trn[eeg].rolling(40).mean()
    trn[eeg] = trn[eeg].rolling(20).mean()
    
    # clip to percentiles
    emax = trn[eeg].quantile(0.998)
    emin = trn[eeg].quantile(0.002)
    trn[eeg] = trn[eeg].where(trn[eeg] <= emax, emax)
    trn[eeg] = trn[eeg].where(trn[eeg] >= emin, emin)
    
trn.describe(percentiles=[0.002,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.998])
trn.hist(['eeg_fp1','eeg_f7','eeg_f8','eeg_t4','eeg_t6','eeg_t5','eeg_t3','eeg_fp2','eeg_o1','eeg_p3','eeg_pz','eeg_f3','eeg_fz','eeg_f4','eeg_c4','eeg_p4','eeg_poz','eeg_c3','eeg_cz','eeg_o2','ecg','r','gsr'])
plt.show()
# plot to compare with before
crw = trn[trn['crew'] == 1] # plot data of just one crew
for eeg in eegs:
    plt.plot(crw.iloc[208000:212000,:][eeg])
plt.show()
# plot ecg
crw = trn[trn['crew'] == 1] # plot data of just one crew
plt.plot(crw.iloc[208000:212000,:]['ecg'])
plt.show()
# do same as with eeg, but no clipping, because noise is now the only issue
# two rounds of smoothing
trn['ecg'] = trn['ecg'].rolling(40).mean()
trn['ecg'] = trn['ecg'].rolling(20).mean()
crw = trn[trn['crew'] == 1] # plot data of just one crew
plt.plot(crw.iloc[208000:212000,:]['ecg'])
plt.show()
# plot r
crw = trn[trn['crew'] == 1] # plot data of just one crew
plt.plot(crw.iloc[208000:212000,:]['r'])
plt.show()
# do same as with eeg, but no clipping, because noise is now the only issue
# two rounds of smoothing
trn['r'] = trn['r'].rolling(40).mean()
trn['r'] = trn['r'].rolling(20).mean()
crw = trn[trn['crew'] == 1] # plot data of just one crew
plt.plot(crw.iloc[208000:212000,:]['r'])
plt.show()
# plot gsr
crw = trn[trn['crew'] == 1] # plot data of just one crew
plt.plot(crw.iloc[208000:212000,:]['gsr'])
plt.show()
# do same as with eeg, but no clipping, because noise is now the only issue
# two rounds of smoothing
trn['gsr'] = trn['gsr'].rolling(40).mean()
trn['gsr'] = trn['gsr'].rolling(20).mean()
crw = trn[trn['crew'] == 1] # plot data of just one crew
plt.plot(crw.iloc[208000:212000,:]['gsr'])
plt.show()
# Now lets scale all values between 0 and 1.
for col in ['eeg_fp1','eeg_f7','eeg_f8','eeg_t4','eeg_t6','eeg_t5',
            'eeg_t3','eeg_fp2','eeg_o1','eeg_p3','eeg_pz','eeg_f3',
            'eeg_fz','eeg_f4','eeg_c4','eeg_p4','eeg_poz','eeg_c3',
            'eeg_cz','eeg_o2','ecg','r','gsr']:
    cmin = trn[col].min()
    cmax = trn[col].max()
    trn[col] = (trn[col]-cmin)/(cmax-cmin)
    
trn.describe(percentiles=[0.02,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.98])
# function to plot eeg data in a certain range
def plotData(start=0, end=-1, bottom=0, top=1):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    crw = trn[trn['crew'] == 1].iloc[start:end,:]
    eegs = [key for key in crw][4:24] # just eeg signals

    # starts and ends of events
    startsEv = {'A':[], 'B':[], 'C':[], 'D':[]}
    endsEv = {'A':[], 'B':[], 'C':[], 'D':[]}
    # starts and ends of experiments
    startsEx = {'CA':[], 'DA':[], 'SS':[]}
    endsEx = {'CA':[], 'DA':[], 'SS':[]}

    # find the starts and ends of all lengths of events and experiments
    curEv = crw.iloc[0,-1]
    curEx = crw.iloc[0,1]
    startsEv[curEv].append(0 + start) # start of first one
    startsEx[curEx].append(0 + start)
    for i in range(1, len(crw)):
        nextEv = crw.iloc[i,-1] # event of next looked at
        nextEx = crw.iloc[i,1]
        if nextEv != curEv:
            endsEv[curEv].append(i + start) # mark end
            startsEv[nextEv].append(i + start) # mark start
            curEv = nextEv
        if nextEx != curEx:
            endsEx[curEx].append(i + start) # mark end
            startsEx[nextEx].append(i + start) # mark start
            curEx = nextEx
    endsEv[curEv].append(len(crw) + start) # put in end of last one
    endsEx[curEx].append(len(crw) + start)

    # show eeg data
    for eeg in eegs:
        ax.plot(crw[eeg])

    # show event spans as transparent backgrounds
    for i in range(len(startsEv['A'])):
        ax.axvspan(startsEv['A'][i], endsEv['A'][i], facecolor='w', alpha=0.7)
    for i in range(len(startsEv['B'])):
        ax.axvspan(startsEv['B'][i], endsEv['B'][i], facecolor='r', alpha=0.7)
    for i in range(len(startsEv['C'])):
        ax.axvspan(startsEv['C'][i], endsEv['C'][i], facecolor='g', alpha=0.7)
    for i in range(len(startsEv['D'])):
        ax.axvspan(startsEv['D'][i], endsEv['D'][i], facecolor='y', alpha=0.7)
        
    # show experiment spans as bar on bottom
    for i in range(len(startsEx['CA'])):
        ax.add_patch(patches.Rectangle((startsEx['CA'][i],bottom), 
                endsEx['CA'][i] - startsEx['CA'][i], (top-bottom)/10, color='g'))
    for i in range(len(startsEx['DA'])):
        ax.add_patch(patches.Rectangle((startsEx['DA'][i],bottom), 
                endsEx['DA'][i] - startsEx['DA'][i], (top-bottom)/10, color='y'))
    for i in range(len(startsEx['SS'])):
        ax.add_patch(patches.Rectangle((startsEx['SS'][i],bottom), 
                endsEx['SS'][i] - startsEx['SS'][i], (top-bottom)/10, color='r'))

    plt.ylim(bottom, top)
    plt.show()
plotData()
plotData(206000,212500)
plotData(113000,115000)
