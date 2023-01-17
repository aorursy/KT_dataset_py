%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
all_paths = [os.path.join(path, file) for path, _, files in os.walk(top = os.path.join('..', 'input')) 
             for file in files if ('.labels' in file) or ('.txt' in file)]
label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
all_files_df = pd.DataFrame({'path': all_paths})
all_files_df['basename'] = all_files_df['path'].map(os.path.basename)
all_files_df['id'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[0])
all_files_df['ext'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[1][1:])
all_files_df.sample(3)
all_training_pairs = all_files_df.pivot_table(values = 'path', 
                                              columns = 'ext', 
                                              index = ['id'], 
                                              aggfunc = 'first').reset_index().dropna()
all_training_pairs
_, test_row = next(all_training_pairs.dropna().tail(1).iterrows())
print(test_row)
read_label_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['class'], index_col = False)
read_xyz_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b'], header = None) #x, y, z, intensity, r, g, b
read_joint_data = lambda c_row, rows: pd.concat([read_xyz_data(c_row['txt'], rows), read_label_data(c_row['labels'], rows)], axis = 1)
read_joint_data(test_row, 10)
%%time
for _, c_row in all_training_pairs.iterrows():
    full_df = read_joint_data(c_row, None)
    with h5py.File('{id}.h5'.format(**c_row), 'w') as h:
        for c_col in full_df.keys():
            h.create_dataset(c_col, data = full_df[c_col].values, compression="gzip", compression_opts=9)
for c_file in glob('*.h5'):
    with h5py.File(c_file) as h:
        print(c_file)
        for k in h.keys():
            print('\t', k, h[k].shape)
!ls -lh *.h5
