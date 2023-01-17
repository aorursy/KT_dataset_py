import numpy as np

from glob import glob

from tqdm import tqdm

import pickle
bpps_matrix = {}

for fname in tqdm(glob('../input/stanford-covid-vaccine/bpps/*.npy')):

    bpps_matrix[fname.split('/')[-1].split('.')[0]] = np.load(fname)
np.savez(open('bpps.npz', 'wb'), **bpps_matrix)
%%time

bpps_dict = np.load('./bpps.npz')

print(bpps_dict['id_00073f8be'])
pickle.dump(bpps_matrix, open('bpps.bin', 'wb'))
%%time

bpps_dict = pickle.load(open('./bpps.bin', 'rb'))

print(bpps_dict['id_00073f8be'])