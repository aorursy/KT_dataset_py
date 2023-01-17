import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from collections import deque

from sklearn.decomposition import FastICA
eeg = pd.read_csv('../input/eegsample/eeg.csv')

# convert from V to uV

eeg *= 10**6 
eeg.iloc[500:2500].plot(figsize=(15,5), legend=False)

plt.xlabel('Time [samples]', fontsize=14, labelpad=10)

plt.ylabel('Voltage [\u03BCV]', fontsize=14)

plt.title('Resting state EEG (63 channels)', fontsize=14)

plt.show()
fig, axs = plt.subplots(2,1, figsize=(15, 7), sharex=True, sharey=True)

axs = axs.ravel()

plt.margins(x=0.001)

fig.add_subplot(111, frameon=False)

plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

axs[0].plot(eeg['Fp1'].iloc[500:2500], label='Fp1', color='rosybrown')

axs[0].legend(loc="upper right", fontsize=12)

axs[1].plot(eeg['Fp2'].iloc[500:2500], label='Fp2', color='silver')

axs[1].legend(loc="upper right", fontsize=12)

plt.xlabel('Time [samples]', fontsize=14, labelpad=15)

plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)

plt.show()
ica = FastICA(n_components=63, random_state=0, tol=0.05)

comps = ica.fit_transform(eeg)
fig, axs = plt.subplots(8,8, figsize=(18, 13), sharex=True, sharey=True)

fig.subplots_adjust(hspace = .4, wspace=0)

axs = axs.ravel()



fig.add_subplot(111, frameon=False)

plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel('Time [samples]', fontsize=14, labelpad=15)



for i in range(63):

    axs[i].plot(comps[1200:1600, i], color='slategrey')

    axs[i].set_title(str(i))
# set artefact components to zero

comps[:,[29,42]] = 0 

restored = ica.inverse_transform(comps)
fig, axs = plt.subplots(2,1, figsize=(15, 7), sharex=True, sharey=True)

axs = axs.ravel()

plt.margins(x=0.001)

fig.add_subplot(111, frameon=False)

plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

axs[0].plot(eeg['Fp1'].iloc[500:2500], label='Fp1_pre', color='rosybrown')

axs[0].plot(np.arange(500,2500), restored[500:2500, 34], label='Fp1_post', color='maroon')

axs[0].legend(loc="upper right", fontsize=12)

axs[1].plot(eeg['Fp2'].iloc[500:2500], label='Fp2_pre', color='silver')

axs[1].plot(np.arange(500,2500), restored[500:2500, 35], label='Fp2_post', color='dimgray')

axs[1].legend(loc="upper right", fontsize=12)

plt.xlabel('Time [samples]', fontsize=14, labelpad=15)

plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)

plt.show()