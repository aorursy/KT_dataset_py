%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
synced_df = pd.read_csv('../input/squat_example.csv')
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

    

    c_ax.plot(synced_df['ML1_TIME'], synced_df['VICON_FHR_{}'.format(ax_name)], label='VICON Knee')

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

                     

        c_ax.plot(synced_df['ML1_TIME'], synced_df['{}{}'.format(var_name,ax_name)]-synced_df['{}{}'.format(var_name,ax_name)].mean(), '-' ,label=var_name, lw=2)

    c_ax.legend()

    c_ax.set_ylabel('{}'.format(ax_name))
fig, (c_ax) = plt.subplots(1, 1, figsize=(20, 10))          

c_ax.plot(synced_df['ML1_TIME'], synced_df['ML1_BELTPACK_ACCEL_X'], '-' ,label='Beltpack IMU')

c_ax.plot(synced_df['ML1_TIME'], synced_df['ML1_ACCEL_X'], '-' ,label='Headset IMU')

c_ax.plot(synced_df['ML1_TIME'], synced_df['FORCE_ACCEL.A2Z'], '-' ,label='Force Plate 2')

c_ax.plot(synced_df['ML1_TIME'], synced_df['FORCE_ACCEL.A6Z'], '-' ,label='Force Plate 6')

c_ax.legend()

c_ax.set_ylabel('Vertical Axis')
fig, (c_ax) = plt.subplots(1, 1, figsize=(35, 10))          

for c_format_tag in ['ML1_BELTPACK_ACCEL_{}', 'ML1_ACCEL_{}', 'FORCE_ACCEL.A2{}', 'FORCE_ACCEL.A6{}']:

    t_acc = np.sqrt(np.sum(np.square(np.stack([synced_df[c_format_tag.format(x)].values for x in 'XYZ'], -1)), -1))

    c_ax.plot(synced_df['ML1_TIME'], t_acc, '-' ,label=c_format_tag.replace('{}','').replace('_',' '))

c_ax.legend()

c_ax.set_ylabel('Total Acceleration')
fig, (c_ax) = plt.subplots(1, 1, figsize=(20, 10))          

for c_format_tag in ['ML1_BELTPACK_ACCEL_{}', 'ML1_ACCEL_{}', 'FORCE_ACCEL.A2{}', 'FORCE_ACCEL.A6{}']:

    t_acc = np.sqrt(np.sum(np.square(np.stack([synced_df[c_format_tag.format(x)].values for x in 'XYZ'], -1)), -1))

    synced_df[c_format_tag.format('total')] = t_acc

    c_ax.plot(synced_df['ML1_TIME'].iloc[500:2500], t_acc[500:2500], '-' ,label=c_format_tag.replace('{}','').replace('_',' '))

c_ax.legend()

c_ax.set_ylabel('Total Acceleration')
from sklearn import linear_model

from sklearn.metrics import r2_score



def corr_lm_plot(in_df, x_col, y_col):

    X = in_df.dropna()[x_col].values.reshape(-1,1)

    y = in_df.dropna()[y_col].values.reshape(-1,1)

    regr = linear_model.LinearRegression()

    regr.fit(X, y)

    print(regr.coef_[0])

    print(regr.intercept_)

    r2 = r2_score(y, regr.predict(X))

    print(r2)

    g = sns.lmplot(x=x_col, y=y_col, data=in_df, aspect=1.0, scatter_kws={'s': 0.5})

    props = dict(boxstyle='round', alpha=0.5,color=sns.color_palette()[0])

    textstr = '$y={1:2.2f}x+{0:2.1f}$\t$R^2={2:2.1%}%$'.format(regr.coef_[0][0], regr.intercept_[0], r2)

    g.ax.text(0.7, 0.1, textstr, transform=g.ax.transAxes, fontsize=14, bbox=props)

corr_lm_plot(synced_df, 'FORCE_ACCEL.A6Z', 'ML1_ACCEL_X')
corr_lm_plot(synced_df, 'FORCE_ACCEL.A6total', 'ML1_ACCEL_total')
corr_lm_plot(synced_df, 'FORCE_ACCEL.A6total', 'ML1_BELTPACK_ACCEL_total')
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
from matplotlib.animation import FuncAnimation

fig, m_axs = plt.subplots(2, 3, figsize=(20, 10))

def draw_points(c_rows, style='.', label=False, alpha=1.0):

    [c_line.remove() 

   for c_ax in m_axs.flatten() 

   for c_line in c_ax.get_lines() 

   if c_line.get_label().startswith('_')];

    for (xy_ax, xz_ax), c_prefix in zip(m_axs.T, 

                                        ['FORCE_ACCEL.A6{}', 'ML1_ACCEL_{}', 'ML1_BELTPACK_ACCEL_{}']):

        xy_ax.set_title(c_prefix.format(''))

        xy_ax.plot(c_rows[c_prefix.format('X')].values,

                  c_rows[c_prefix.format('Y')].values,

                   style,

                   alpha=alpha,

                   label=c_prefix if label else None

                  )

        xy_ax.set_xlabel('X')

        xy_ax.set_ylabel('Y')

        xz_ax.plot(c_rows[c_prefix.format('X')].values,

                  c_rows[c_prefix.format('Z')].values,

                   style,

                   alpha=alpha,

                   label=c_prefix if label else None

                  )

        xz_ax.set_xlabel('X')

        xz_ax.set_ylabel('Z')

draw_points(synced_df, '-', label=True, alpha=0.25)

out_anim = FuncAnimation(fig, func=draw_points, frames=[c_rows for _, c_rows in synced_df.groupby(synced_df.index//125)])
from IPython.display import HTML

#HTML(out_anim.to_jshtml(fps=3))
out_anim.save('imu_vs_force_plate.gif', bitrate=8000, fps=8)