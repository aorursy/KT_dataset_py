%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
all_paths = [os.path.join(path, file) for path, _, files in os.walk(top = os.path.join('..', 'input')) 
             for file in files if ('.labels' in file) or ('.txt' in file)]
label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
all_files_df = pd.DataFrame({'path': all_paths})
all_files_df['basename'] = all_files_df['path'].map(os.path.basename)
all_files_df['id'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[0])
all_files_df['ext'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[1][1:])
all_files_df.sample(3)
all_training_pairs = all_files_df.pivot_table(values = 'path', columns = 'ext', index = ['id'], aggfunc = 'first').reset_index()
all_training_pairs
_, test_row = next(all_training_pairs.dropna().tail(1).iterrows())
print(test_row)
read_label_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['class'], index_col = False)
read_xyz_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b'], header = None) #x, y, z, intensity, r, g, b
read_joint_data = lambda c_row, rows: pd.concat([read_xyz_data(c_row['txt'], rows), read_label_data(c_row['labels'], rows)], axis = 1)
read_joint_data(test_row, 10)
%%time
full_df = read_joint_data(test_row, None)
test_df = full_df[(full_df.index % 10)==0]
print(full_df.shape[0], 'rows', test_df.shape[0], 'number of filtered rows')
%%time
fig, m_axs = plt.subplots(1, 3, figsize = (20, 5))
ax_names = 'xyz'
for i, c_ax in enumerate(m_axs.flatten()):
    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]
    c_ax.scatter(test_df[plot_axes[0]],
                test_df[plot_axes[1]],
                c=test_df[['r', 'g', 'b']].values/255, 
                 s=1
                )
    c_ax.set_xlabel(plot_axes[0])
    c_ax.set_ylabel(plot_axes[1])
%%time
max_keys = max(label_names.keys())
fig, m_axs = plt.subplots(max_keys+1, 3, figsize = (20, 30))
for i, c_axs in enumerate(m_axs.T):
    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]
    for c_ax, (c_key, c_value) in zip(c_axs, label_names.items()):
        c_df = test_df[test_df['class']==c_key]
        c_ax.plot(c_df[plot_axes[0]],
                  c_df[plot_axes[1]],
                  '.', 
                  label = c_value
                 )
        c_ax.set_title('{}: {}'.format(c_value, ''.join(plot_axes)))
        c_ax.set_xlabel(plot_axes[0])
        c_ax.set_ylabel(plot_axes[1])
        c_ax.axis('off')
fig, m_axs = plt.subplots(1, 3, figsize = (30, 10))
for i, c_ax in enumerate(m_axs.T):
    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]
    for (c_key, c_value) in label_names.items():
        c_df = test_df[test_df['class']==c_key]
        c_ax.plot(c_df[plot_axes[0]],
                    c_df[plot_axes[1]],
                '.', 
                  label = c_value
                    )
    c_ax.legend()
    c_ax.set_xlabel(plot_axes[0])
    c_ax.set_ylabel(plot_axes[1])
test_df = full_df[(full_df.index % 50)==0]
print(full_df.shape[0], 'rows', test_df.shape[0], 'number of filtered rows')
%%time
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.scatter(
            test_df['x'], test_df['y'], test_df['z'],
            c=test_df[['r', 'g', 'b']].values/255, s=3)  
ax.view_init(15, 165)
ax.view_init(45, 220)
fig.savefig('3D_rendering.png', dpi = 300)
fig
%%time
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')  
for (c_key, c_value) in label_names.items():
    c_df = test_df[test_df['class']==c_key]
    ax.plot(c_df['x'], c_df['y'], c_df['z'], '.', label = c_value, alpha = 0.5)  
ax.legend()
ax.view_init(15, 165)
fig.savefig('3d_labels.png', dpi = 300)
ax.view_init(45, 220)
fig
%%time
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.scatter(test_df['x'], test_df['y'], test_df['z'],
           c=test_df[['r', 'g', 'b']].values/255, s=3)  
for (c_key, c_value) in label_names.items():
    if c_key>0:
        c_df = test_df[test_df['class']==c_key]
        ax.plot(c_df['x'], c_df['y'], c_df['z'], '.', label = c_value, alpha = 0.25, markersize = 1.0)  
ax.legend()
ax.view_init(45, 220)
fig.savefig('3d_labels_overlay.png', dpi = 600)
