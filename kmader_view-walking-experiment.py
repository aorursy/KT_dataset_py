%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
synced_df = pd.read_csv('../input/example.csv')
all_acc = synced_df[['ML1_BELTPACK_ACCEL_X', 'ML1_BELTPACK_ACCEL_Y','ML1_BELTPACK_ACCEL_Z']].values

acc_scalar = 9.8/np.nanmedian(np.sqrt(np.sum(np.square(all_acc), 1)))

trans_array = np.array([[-95,   7, -28],[-29, -28,  91],[ -1,  95,  29]])/100.0

trans_array[:, 1] *=-1

#trans_array = np.eye(3)

trans_bp_imu = np.matmul(all_acc*acc_scalar, trans_array)

# replace array

synced_df['ML1_BELTPACK_ACCEL_X'] = trans_bp_imu[:, 2]

synced_df['ML1_BELTPACK_ACCEL_Y'] = trans_bp_imu[:, 1]

synced_df['ML1_BELTPACK_ACCEL_Z'] = trans_bp_imu[:, 0]
for k in synced_df.columns:

    if k.startswith('ML1'):

        pass

        print(k)
synced_df.describe()
a_mass = 66/2 #kg per foot

for i in range(1, 7):

    for k in 'XYZ':

        synced_df['FORCE_ACCEL.A{}{}'.format(i, k)] = synced_df['FORCE_FORCE.F{}{}'.format(i, k)]/a_mass
synced_df.describe().T
fig, m_axs = plt.subplots(3, 1, figsize=(15, 10))

for c_ax, ax_name in zip(m_axs, 'XYZ'):

    c_ax.plot(synced_df['ML1_TIME'], 1000*synced_df['ML1_TRANSLATION_{}'.format(ax_name)], '-.' ,label='ML Headpose', lw=2)

    c_ax.plot(synced_df['ML1_TIME'], 1000*synced_df['ML1_TOTEM_POSE_TRANSLATION_{}'.format(ax_name)], '-.' ,label='ML Totem', lw=2)

    c_ax.plot(synced_df['ML1_TIME'], synced_df['VICON_HEADPOSE_{}'.format(ax_name)], label='VICON Googles')

    c_ax.plot(synced_df['ML1_TIME'], synced_df['VICON_BELT_{}'.format(ax_name)], label='VICON Beltpack')

    c_ax.plot(synced_df['ML1_TIME'], synced_df['VICON_FOOT_LF_{}'.format(ax_name)], label='VICON Foot')

    c_ax.legend()

    c_ax.set_ylabel('{} (mm)'.format(ax_name))
fig, m_axs = plt.subplots(6, 1, figsize=(40, 20))

for c_ax, var_name in zip(m_axs, [

    'ML1_BELTPACK_ACCEL_',  #beltpack

    'ML1_ACCEL_', # headset

    'ML1_BELTPACK_GYR_',

    'ML1_OMEGA_'

    ]+

     ['FORCE_ACCEL.A{}'.format(i) for i in [2, 6]]):

    for ax_name in 'XYZ':

        c_ax.plot(synced_df['ML1_TIME'], synced_df['{}{}'.format(var_name, ax_name)], label=ax_name)

    c_ax.legend()

    c_ax.set_ylabel(var_name)
fig, m_axs = plt.subplots(3, 1, figsize=(15, 10))

for c_ax, ax_name in zip(m_axs, 'XYZ'):

    for var_name in ['ML1_BELTPACK_ACCEL_', 'ML1_ACCEL_']+['FORCE_ACCEL.A{}'.format(i) for i in [2, 6]]:

                     

        c_ax.plot(synced_df['ML1_TIME'], synced_df['{}{}'.format(var_name,ax_name)], '-' ,label=var_name, lw=2)

    c_ax.legend()

    c_ax.set_ylabel('{}'.format(ax_name))
fig, (c_ax) = plt.subplots(1, 1, figsize=(20, 10))          

c_ax.plot(synced_df['ML1_TIME'], synced_df['ML1_BELTPACK_ACCEL_Z'], '-' ,label='Beltpack IMU')

c_ax.plot(synced_df['ML1_TIME'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU')

c_ax.plot(synced_df['ML1_TIME'], synced_df['FORCE_ACCEL.A2Z'], '-' ,label='Force Plate 2')

c_ax.plot(synced_df['ML1_TIME'], synced_df['FORCE_ACCEL.A6Z'], '-' ,label='Force Plate 6')

c_ax.legend()

c_ax.set_ylabel('Vertical Axis')
fig, (c_ax, d_ax) = plt.subplots(1, 2, figsize=(20, 10))          

c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['ML1_BELTPACK_ACCEL_Z'], '.-' ,label='Beltpack IMU')

c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['ML1_ACCEL_X'], '.-' ,label='Headset IMU')

c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['FORCE_ACCEL.A6Z'], '.-' ,label='Force Plate 6')

c_ax.legend()

c_ax.set_xlabel('Force Plate 2')





d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['ML1_BELTPACK_ACCEL_Z'], '.-' ,label='Beltpack IMU')

d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['ML1_ACCEL_X'], '.-' ,label='Headset IMU')

d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['FORCE_ACCEL.A2Z'], '.-' ,label='Force Plate 2')

d_ax.legend()

d_ax.set_xlabel('Force Plate 6')
fig, (c_ax, d_ax) = plt.subplots(1, 2, figsize=(20, 10))          



c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU', alpha=0.25)

c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['FORCE_ACCEL.A6Z'], '-' ,label='Force Plate 6')





d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU')

d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['FORCE_ACCEL.A2Z'], '-' ,label='Force Plate 2')



for ax_name in 'XYZ':

    c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['ML1_BELTPACK_ACCEL_{}'.format(ax_name)], '.' ,label=f'Beltpack IMU {ax_name}', alpha=0.25)

    d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['ML1_BELTPACK_ACCEL_{}'.format(ax_name)], '.' ,label=f'Beltpack IMU {ax_name}', alpha=0.25)

    

c_ax.legend()

c_ax.set_xlabel('Force Plate 2')

d_ax.legend()

d_ax.set_xlabel('Force Plate 6')
fig, (c_ax) = plt.subplots(1, 1, figsize=(20, 10))          

c_ax.plot(synced_df['ML1_TIME'], 10+0.5*synced_df['ML1_BELTPACK_ACCEL_Z'].shift(int(250*0.45)), '-' ,label='Beltpack IMU')

c_ax.plot(synced_df['ML1_TIME'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU')

c_ax.plot(synced_df['ML1_TIME'], synced_df['FORCE_ACCEL.A2Z'], '-' ,label='Force Plate 2')

c_ax.plot(synced_df['ML1_TIME'], synced_df['FORCE_ACCEL.A6Z'], '-' ,label='Force Plate 6')

c_ax.legend()

c_ax.set_ylabel('Vertical Axis')
fig, (c_ax, d_ax) = plt.subplots(1, 2, figsize=(20, 10))          



#c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU', alpha=0.25)

#c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['FORCE_ACCEL.A6Z'], '-' ,label='Force Plate 6')





#d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU')

#d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['FORCE_ACCEL.A2Z'], '-' ,label='Force Plate 2')



for ax_name in 'XYZ':

    c_ax.plot(synced_df['FORCE_ACCEL.A2Z'], synced_df['ML1_BELTPACK_ACCEL_{}'.format(ax_name)].shift(int(250*0.45)), '.' ,label=f'Beltpack IMU {ax_name}', alpha=0.25)

    d_ax.plot(synced_df['FORCE_ACCEL.A6Z'], synced_df['ML1_BELTPACK_ACCEL_{}'.format(ax_name)].shift(int(250*0.45)), '.' ,label=f'Beltpack IMU {ax_name}', alpha=0.25)

    

c_ax.legend()

c_ax.set_xlabel('Force Plate 2')

d_ax.legend()

d_ax.set_xlabel('Force Plate 6')