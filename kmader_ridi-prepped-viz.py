%matplotlib inline

import numpy as np

import pandas as pd

from pathlib import Path

from scipy.integrate import cumtrapz

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
ridi_df = pd.read_csv('../input/ridi-generate-data/all_readings.csv.zip', compression=None)

ridi_df.head(5)
sample_df = ridi_df[(ridi_df['activity']==ridi_df['activity'].iloc[0]) & (ridi_df['person']==ridi_df['person'].iloc[0])].copy()

sample_df['timestamp_s'] = (sample_df['time']-sample_df['time'].min())/1e9

sample_df = sample_df.query('timestamp_s<50').copy() # just a short time window

sample_df.shape[0]
sample_df.plot('timestamp_s', ['pos_x', 'pos_y', 'pos_z'])
sample_df.plot('timestamp_s', ['acce_x', 'acce_y', 'acce_z'])
for c_x in 'xyz':

    grav_1 = sample_df['acce_{}'.format(c_x)].rolling(200, center=True).median().bfill().ffill()

    grav_2 = grav_1.rolling(50, center=True).mean().bfill().ffill()

    sample_df['gravity_{}'.format(c_x)] = grav_2

grav_scalar = 9.81/np.linalg.norm(sample_df[['gravity_x', 'gravity_y', 'gravity_z']].values, axis=1)

for c_x in 'xyz':

    sample_df['gravity_{}'.format(c_x)] *= grav_scalar

sample_df.plot('timestamp_s', ['gravity_x', 'gravity_y', 'gravity_z'])
for c_x in 'xyz':    

    sample_df['lin_acce_{}'.format(c_x)] = sample_df['acce_{}'.format(c_x)]-sample_df['gravity_{}'.format(c_x)]
pose_start = sample_df.iloc[0] # for the initial conditions

for c_x in 'xyz':

    sample_df['ivel_{}'.format(c_x)] = cumtrapz(sample_df['lin_acce_{}'.format(c_x)].values, 

                                           x=sample_df['timestamp_s'].values, 

                                           initial=0)

    sample_df['ipos_{}'.format(c_x)] = cumtrapz(sample_df['ivel_{}'.format(c_x)].values, 

                                           x=sample_df['timestamp_s'], 

                                           initial=pose_start['pos_{}'.format(c_x)])
fig, m_axs = plt.subplots(3, 1, figsize=(20, 12))

for c_x, c_ax in zip('xyz', m_axs):

    c_ax.plot(sample_df['timestamp_s'], sample_df['lin_acce_{}'.format(c_x)], '.', label='Linear Acceleration')

    c_ax.plot(sample_df['timestamp_s'], sample_df['ivel_{}'.format(c_x)], label='Integrated Velocity')

    c_ax.plot(sample_df['timestamp_s'], sample_df['ipos_{}'.format(c_x)], label='Integrated Position')

    c_ax.plot(sample_df['timestamp_s'], sample_df['pos_{}'.format(c_x)], '-', label='Actual Pose')

    c_ax.legend()

    c_ax.set_title(c_x)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

ax1.plot(sample_df['ipos_x'], sample_df['ipos_y'], '.-', label='Integrated Position')

ax1.plot(sample_df['pos_x'], sample_df['pos_y'], '+-', label='Actual Pose')

ax1.legend()

ax1.axis('equal');
fig = plt.figure(figsize=(10, 10), dpi=300)

ax1 = fig.add_subplot(111, projection='3d')

ax1.plot(sample_df['ipos_x'], sample_df['ipos_y'], sample_df['ipos_z'], '.-', label='Integrated Position')

ax1.plot(sample_df['pos_x'], sample_df['pos_y'], sample_df['pos_z'], '.-', label='Actual Position')

ax1.legend()

ax1.axis('equal');

fig.savefig('hr_img.png')
pose_start = sample_df.iloc[0] # for the initial conditions

for c_x in 'xyz':

    vel_vec = cumtrapz(sample_df['lin_acce_{}'.format(c_x)].values, 

                                           x=sample_df['timestamp_s'].values, 

                                           initial=0)

    for i in range(0, sample_df.shape[0], 200): # once a second

        vel_vec[(i+1):] -= vel_vec[i]

    sample_df['ivel_{}'.format(c_x)] = vel_vec

    sample_df['ipos_{}'.format(c_x)] = cumtrapz(sample_df['ivel_{}'.format(c_x)].values, 

                                           x=sample_df['timestamp_s'], 

                                           initial=pose_start['pos_{}'.format(c_x)])

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

ax1.plot(sample_df['ipos_x'], sample_df['ipos_y'], '.-', label='Integrated Position')

ax1.plot(sample_df['pos_x'], sample_df['pos_y'], '+-', label='Actual Pose')

ax1.legend()

ax1.axis('equal');
fig = plt.figure(figsize=(10, 10), dpi=300)

ax1 = fig.add_subplot(111, projection='3d')

ax1.plot(sample_df['ipos_x'], sample_df['ipos_y'], sample_df['ipos_z'], '.-', label='Integrated Position')

ax1.plot(sample_df['pos_x'], sample_df['pos_y'], sample_df['pos_z'], '.-', label='Actual Position')

ax1.legend()

ax1.axis('equal');

fig.savefig('hr_img.png')

sample_df = sample_df.query('timestamp_s<10').copy()
diff = lambda x: x.diff()

smooth_diff =  lambda x, n=100: x.rolling(n, center=True).mean().bfill().ffill().diff() if n>0 else diff(x)
for c_x in 'xyz':

    sample_df['vel_from_pos_{}'.format(c_x)] = smooth_diff(sample_df['pos_{}'.format(c_x)], n=10)/smooth_diff(sample_df['timestamp_s'])

    sample_df['acc_from_vel_{}'.format(c_x)] = smooth_diff(sample_df['vel_from_pos_{}'.format(c_x)], n=10)/smooth_diff(sample_df['timestamp_s'])
fig, m_axs = plt.subplots(3, 1, figsize=(20, 12))

for c_x, c_ax in zip('xyz', m_axs):

    c_ax.plot(sample_df['timestamp_s'], sample_df['lin_acce_{}'.format(c_x)], '.', label='IMU Acceleration')

    c_ax.plot(sample_df['timestamp_s'], sample_df['acc_from_vel_{}'.format(c_x)], label='Derivative Acceleration')

    c_ax.legend()

    c_ax.set_title(c_x)