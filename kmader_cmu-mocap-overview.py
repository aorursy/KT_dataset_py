!pip install -qq c3d
%matplotlib inline

import c3d

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = Path('..') / 'input'

datasets_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/subjects/*/*.c3d'))})

datasets_df['Subject'] = datasets_df['path'].map(lambda x: x.parent.stem)

datasets_df['Activity'] = datasets_df['path'].map(lambda x: x.stem.split('_')[-1].lower())

datasets_df.sample(3)
test_rec = datasets_df.iloc[1]

print(test_rec)

test_file = test_rec['path'] 


# preview what is in the file a bit

with test_file.open('rb') as hf:

    reader = c3d.Reader(hf)

    print('Frames:', len(list(reader.read_frames())))

    for i, points, analog in reader.read_frames():

        print('frame {}: point {}, analog {}'.format(

            i, points.shape, analog.shape))

        if i>5:

            break
with test_file.open('rb') as hf:

    all_fields = []

    reader = c3d.Reader(hf)

    scale_xyz = np.abs(reader.point_scale) # don't flip everything

    for frame_no, points, _ in reader.read_frames(copy=False):

        for (x, y, z, err, cam), label in zip(points, 

                                     reader.point_labels):

            c_field = {'frame': frame_no, 

                       'time': frame_no / reader.point_rate,

                       'point_label': label.strip()}

            c_field['x'] = scale_xyz*x

            c_field['y'] = scale_xyz*y

            c_field['z'] = scale_xyz*z

            c_field['err'] = err<0

            c_field['cam'] = cam<0

            all_fields += [c_field]

all_df = pd.DataFrame(all_fields)[['time', 'point_label', 'x', 'y', 'z',  'cam', 'err', 'frame']]

all_df.sample(5)
all_df[['cam', 'err', 'point_label']].describe()
fig, m_axs = plt.subplots(3, 1, figsize=(20, 20))

for c_x, c_ax in zip('xyz', m_axs):

  for c_grp, c_rows in all_df.groupby('point_label'):

    c_rows = c_rows.copy().query('not err')

    c_ax.plot(c_rows['time'], c_rows[c_x], label=c_grp)

  c_ax.set_title(c_x);

#m_axs[0].legend()
fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(111, projection='3d')

for c_grp, c_rows in all_df.query('not err').groupby('point_label'):

  ax1.plot(c_rows['x'], c_rows['y'], c_rows['z'], label=c_grp)

valid_df = all_df.query('not err')

fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(111, projection='3d')

for c_grp, c_rows in valid_df.groupby('point_label'):

  ax1.plot(c_rows['y'], c_rows['x'], c_rows['z'], '-', label=c_grp)

#ax1.legend()

ax1.set_xlabel('x')

ax1.set_ylabel('y')

ax1.set_zlabel('z')

#ax1.set_aspect('equal')
valid_df = all_df.query('not err')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

for c_grp, c_rows in valid_df.groupby('point_label'):

  ax1.plot(c_rows['x'], c_rows['z'], '-', label=c_grp)

  ax1.axis('equal')

  ax2.plot(c_rows['y'], c_rows['z'], '-', label=c_grp)



ax2.legend()

ax1.set_xlabel('x')

ax1.set_ylabel('z')





ax2.set_xlabel('y')

ax2.set_ylabel('z')

#ax1.legend()
from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from IPython.display import FileLink

from matplotlib.animation import FuncAnimation



valid_df = all_df.query('not err')

valid_df['y']*=1



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))



for c_col, (c_grp, c_rows) in zip(cycle(colors), valid_df.groupby('point_label')):

  ax1.plot(c_rows['x'], c_rows['z'], '-', label=c_grp, alpha=0.25, color=c_col)

  ax1.axis('equal')

  ax2.plot(c_rows['y'], c_rows['z'], '-', label=c_grp, alpha=0.25, color=c_col)





ax1.set_xlabel('x')

ax1.set_ylabel('z')





ax2.set_xlabel('y')

ax2.set_ylabel('z')

frame_data = [c_df for _, c_df in valid_df.groupby(pd.qcut(valid_df['time'], 20))]

def update_frame(in_df):

  [(f.set_alpha(0.1), f.set_marker('None')) for f in ax1.get_lines()]

  for c_col, (c_grp, c_rows) in zip(cycle(colors), in_df.groupby('point_label')):

    marker = '.'

    alpha = 0.25

    

    if c_grp.endswith('HH') or c_grp.endswith('HL'):

      marker='s'

      alpha = 0.6

    

    ax1.plot(c_rows['x'], c_rows['z'], marker, label=c_grp, alpha=alpha, color=c_col)

    ax2.plot(c_rows['y'], c_rows['z'], marker, label=c_grp, alpha=alpha, color=c_col)



out_path = 'simple_animation.gif'

FuncAnimation(fig, update_frame, frame_data).save(out_path, 

                                                  bitrate=8000,

                                                  fps=8)

plt.close('all')

FileLink(out_path)
heel_df = valid_df[valid_df['point_label'].map(lambda x: x.endswith('HEE') or x.endswith('TOE'))].copy()

fig, m_axs = plt.subplots(3, 1, figsize=(20, 10))

for c_x, c_ax in zip('xyz', m_axs):

  for c_grp, c_rows in heel_df.groupby('point_label'):

    c_ax.plot(c_rows['time'], c_rows[c_x], label=c_grp)

  c_ax.set_title(c_x);

m_axs[0].legend()