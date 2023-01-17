import os
import numpy as np
import keras
import soundfile as sf
%load_ext memory_profiler
# load sounds
train_dir = '../input/songs/songs'
paths = [os.path.join(train_dir, x) for x in os.listdir(train_dir)[:10]]
dataset = []
for p in paths:
    audio, _ = sf.read(p)
    dataset.append(audio)
# pad sequences to uniform length
%memit dataset_pad_sequences = keras.preprocessing.sequence.pad_sequences([p.tolist() for p in dataset])
dataset_pad_sequences.shape
max_length = max([len(s) for s in dataset])
max_length
# TODO try http://devdocs.io/numpy~1.13/generated/numpy.pad to avoid converting numpy array to list and back
%memit np.pad(dataset[1], max_length, 'constant', constant_values=0)