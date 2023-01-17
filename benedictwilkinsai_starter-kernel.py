def hgallery(x, n=None): #util for showing images

    if n is None:

        n = images.shape[0]

    m,h,w,c = x.shape

    n = min(m, n) #if n is larger, just use m

    if m % n != 0:

        pad = ((0, n - (m % n)),*([(0,0)]*(len(x.shape)-1)))

        x = np.pad(x, pad)

        m,h,w,c = x.shape

    return x.swapaxes(1,2).reshape(m//n, w * n, h, c).swapaxes(1,2)
import numpy as np 

import h5py

import matplotlib.pyplot as plt



path = "/kaggle/input/atari-anomaly-dataset-aad/AAD/clean/BreakoutNoFrameskip-v4/episode(10).hdf5"

file = h5py.File(path, 'r')

state = file['state'][...]

action = file['action'][...]



print(state.shape, state.dtype)

print(action.shape, action.dtype)



gallery = hgallery(state, n=10)

plt.figure(figsize = (20,3))

plt.imshow(gallery[0])
import json

import numpy as np 

import h5py

import matplotlib.pyplot as plt



path_meta = "/kaggle/input/atari-anomaly-dataset-aad/AAD/anomaly/BreakoutNoFrameskip-v4/meta.json"



#read meta

with open(path_meta) as json_file:

    meta = json.load(json_file)

    for k,v in meta['anomaly'].items():

        print("{0:<20} : {1} ...".format(k,", ".join(v[:4])))



path = "/kaggle/input/atari-anomaly-dataset-aad/AAD/anomaly/BreakoutNoFrameskip-v4/episode.hdf5"

file = h5py.File(path, 'r')

state = file['state'][...]   #images in NHWC (uint8) format

action = file['action'][...] #actions N (uint8)

label = file['label'][...]   #state labels N (uint8)

tlabel = file['tlabel'][...] #transition labels N (uint8)



print(state.shape, state.dtype)

print(action.shape, action.dtype)

print(label.shape, label.dtype)

print(tlabel.shape, tlabel.dtype)



gallery = hgallery(state, n=10)

plt.figure(figsize = (20,3))

plt.imshow(gallery[0])
