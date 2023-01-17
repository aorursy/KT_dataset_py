import pandas as pd
import librosa
import matplotlib.pylab as plt
import numpy as np
import random
from scipy import signal
import pickle
%matplotlib inline
df = pd.read_csv("../input/dolphin-vocalizations/signature_whistle_metadata.csv")
signature_whistles = pickle.load(open("../input/dolphin-vocalizations/signature_whistles.p", "rb"))
df = df.loc[df.ID1 == df.ID2]
df = df.drop(166) # This one is wonky
signature_whistles = [signature_whistles[i] for i in df.index]
df.reset_index(inplace=True)
# Get list of dolphin names
dolphins = df.ID1.unique()
# Create spectrograms
sampling_rate = 48000
spectrograms = [None] * len(df)

for i, whistle in enumerate(signature_whistles):
    spectrograms[i] = signal.spectrogram(whistle, sampling_rate)
def plot_sig_whistles_spectogram(dolphin, window_size=None):
    
    n = 21

    random.seed(1)
    exemplars = [spectrograms[i] for i in random.choices(df.loc[df.ID1 == dolphin].index, k=n)]
   
    fig, axes = plt.subplots(7,3, figsize=(15,15))
    for i in range(7):
        for j in range(3):
            f, t, sxx = exemplars[i*3+j]
            axes[i,j].pcolormesh(t, f, sxx)
    fig.suptitle('Signature Whistle Spectrogram for dolphin ' + dolphin, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

for dolphin in dolphins: plot_sig_whistles_spectogram(dolphin)
def plot_sig_whistles_waveform(dolphin, same_scale=True):
    
    n = 21

    random.seed(1)
    idx = random.choices(df.loc[df.ID1 == dolphin].index, k=n)
    sigs = [signature_whistles[i] for i in idx]
    
    max_val = 0
    min_val = 0
    for i in range(n):
        # print(sigs[i])
        val = np.max(sigs[i])
        # print(val)
        if val > max_val: max_val = val
        val = np.min(sigs[i])
        if val < min_val: min_val = val

    fig, axes = plt.subplots(7,3, figsize=(15,15))  
    for i in range(7):
        for j in range(3):
            axes[i,j].plot(sigs[i*3+j])
            if same_scale: axes[i,j].set_ylim([min_val,max_val])
            lbl = "D:" + str(df.loc[idx[i*3+j],'day']) + "; Sess:" + str(df.loc[idx[i*3+j],'session']) + "; Ch:" + str(df.loc[idx[i*3+j],'channel'])
            axes[i,j].set_title(lbl + "; Start: " + df.loc[idx[i*3+j],'start_time'] + "; End: " + df.loc[idx[i*3+j],'end_time'])
    fig.suptitle('Signature Whistle Waveforms for dolphin ' + dolphin, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

for dolphin in dolphins: plot_sig_whistles_waveform(dolphin, same_scale=True)
