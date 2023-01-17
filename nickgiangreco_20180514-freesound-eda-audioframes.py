import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/train.csv")
print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.label.unique()))
test = pd.read_csv("../input/sample_submission.csv")
_,ax=plt.subplots(figsize=(7,7))
data = train.groupby(['label']).size().sort_values()
p = data.plot(kind='barh',ax=ax)
p = ax.set_xlabel("Count")
p = ax.set_ylabel("Audio class")
import IPython.display as ipd  # To play sound in the notebook
fname = '../input/audio_train/' + '00044347.wav'   # Hi-hat
ipd.Audio(fname)
import os
train_files = os.listdir("../input/audio_train")
train_files_dict = dict(zip(train_files,range(len(train_files))))
sorted_train = train.iloc[train.fname.map(train_files_dict)]
import ipywidgets as widget
labels = train['label'].unique()
w = widget.Dropdown(options = labels, value=labels[0])
w
train_files_inds = train['label'] == w.label
train['fname'][train_files_inds].head(5)
from scipy.io import wavfile
path = "../input/audio_train/"
fname = train['fname'][train_files_inds].values[0]
rate, data = wavfile.read(path + fname)
plt.plot(data, '-', );
seed = 2
np.random.seed(seed)
show_df = train.query('manually_verified == 1').sort_values('label')
labels = show_df['label'].unique()

for label in labels[:5]:
    
    train_files_inds = show_df['label'] == label

    rand_inds = np.random.randint(0,show_df['fname'][train_files_inds].count(),5)
    fnames = show_df['fname'].iloc[rand_inds]

    _, axs = plt.subplots(figsize=(17,4),nrows=1,ncols=5)

    for i,fname in enumerate(fnames):
        rate, data = wavfile.read(path + fname)
        axs[i].plot(data, '-', label=fname)
        axs[i].legend()
    plt.suptitle(label,x=0.04,y=0.5,horizontalalignment='center', fontsize=20)
    del rate
    del data
del axs
arrs = []
lens = [5]*5
for i in lens:
    arrs.append(np.random.randint(7,size=i))
arrs = np.array(arrs)
arrs
np.median(arrs,axis=1)
arrs = []
lens = [5,10,7,6,9]
for i in lens:
    arrs.append(np.random.randint(7,size=i))
arrs = np.array(arrs)
arrs
maxarrn = np.max([len(i) for i in arrs])
padded_arrs = [np.pad(arr,(0,maxarrn-len(arr)),mode='constant')for arr in arrs]
padded_arrs = np.array(padded_arrs)
display(padded_arrs)

med_padded_arrs = np.median(padded_arrs,axis=0)
med_padded_arrs
dists = [np.linalg.norm(np.pad(arr,(0,len(med_padded_arrs)-len(arr)),mode='constant')-med_padded_arrs) for arr in arrs]
dists = np.array(dists)
dists
dists.argsort()[::-1]
label = 'Acoustic_guitar'
sub = show_df[show_df.label==label]
fnames = sub.fname.values
arrs = []
for i,fname in enumerate(fnames):
        rate, data = wavfile.read(path + fname)
        arrs.append(data)
maxarrn = np.max([len(i) for i in arrs])
padded_arrs = [np.pad(arr,(0,maxarrn-len(arr)),mode='constant')for arr in arrs]
padded_arrs = np.array(padded_arrs)
med_padded_arrs = np.median(padded_arrs,axis=0)
dists = [np.linalg.norm(np.pad(arr,(0,len(med_padded_arrs)-len(arr)),mode='constant')-med_padded_arrs) for arr in arrs]
dists = np.array(dists)
fnames = sub.iloc[dists.argsort()[::-1]].head(5).fname.values
_, axs = plt.subplots(figsize=(17,4),nrows=1,ncols=5)
for i,fname in enumerate(fnames):
    rate, data = wavfile.read(path + fname)
    axs[i].plot(data, '-', label=fname)
    axs[i].legend()
plt.suptitle(label,x=0.04,y=0.5,horizontalalignment='center', fontsize=20)
show_df = train.query('manually_verified == 1').sort_values('label')
labels = show_df['label'].unique()

for label in labels:
    sub = show_df[show_df.label==label]
    fnames = sub.fname.values
    arrs = []
    for i,fname in enumerate(fnames):
            rate, data = wavfile.read(path + fname)
            arrs.append(data)
    maxarrn = np.max([len(i) for i in arrs])
    padded_arrs = [np.pad(arr,(0,maxarrn-len(arr)),mode='constant')for arr in arrs]
    padded_arrs = np.array(padded_arrs)
    med_padded_arrs = np.median(padded_arrs,axis=0)
    dists = [np.linalg.norm(np.pad(arr,(0,len(med_padded_arrs)-len(arr)),mode='constant')-med_padded_arrs) for arr in arrs]
    dists = np.array(dists)
    fnames = sub.iloc[dists.argsort()[::-1]].head(5).fname.values
    _, axs = plt.subplots(figsize=(17,4),nrows=1,ncols=5)
    for i,fname in enumerate(fnames):
        rate, data = wavfile.read(path + fname)
        axs[i].plot(data, '-', label=fname)
        axs[i].legend()
    plt.suptitle(label,x=0.04,y=0.5,horizontalalignment='center', fontsize=20)
del axs
fname = '../input/audio_train/' + '991fa1d7.wav'   # Hi-hat
ipd.Audio(fname)
