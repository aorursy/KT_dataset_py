import numpy as np

from scipy import ndimage as nd

from matplotlib import pyplot as plt
!wget https://github.com/jeanpat/DeepFISH/blob/master/dataset/Cleaned_FullRes_2164_overlapping_pairs.npz?raw=true

!mv Cleaned_FullRes_2164_overlapping_pairs.npz?raw=true Clean2164.npz
# There's a trick to load and uncompress a numpy .npz array

# https://stackoverflow.com/questions/18231135/load-compressed-data-npz-from-file-using-numpy-load/44693995

#

dataset = np.load('Clean2164.npz')

data = dataset.f.arr_0

data.shape
N=203

plt.figure(figsize=(10,8))

plt.subplot(121)

plt.imshow(data[N,:,:,0], cmap=plt.cm.gray)

plt.subplot(122)

plt.imshow(data[N,:,:,1], cmap=plt.cm.flag_r)
!wget https://github.com/jeanpat/DeepFISH/blob/master/dataset/Cleaned_FullRes_2164_overlapping_pairs.h5?raw=true

!mv Cleaned_FullRes_2164_overlapping_pairs.h5?raw=true Clean2164.h5
import h5py

filename = './Clean2164.h5'

h5f = h5py.File(filename,'r')

pairs = h5f['chroms_data'][:]

h5f.close()

print('dataset is a numpy array of shape:', pairs.shape)