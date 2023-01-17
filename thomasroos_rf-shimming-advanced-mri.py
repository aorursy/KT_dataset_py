# List available input files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Read .TSV files from I'TIS

df_15T = pd.read_csv('/kaggle/input/dielectricpropertiesmr/15T.tsv', delimiter='\t')

df_30T = pd.read_csv('/kaggle/input/dielectricpropertiesmr/30T.tsv', delimiter='\t')

df_70T = pd.read_csv('/kaggle/input/dielectricpropertiesmr/70T.tsv', delimiter='\t')



# Print example (7.0T) dataframe:

print('Dielectrical properties at 7T')

print(df_70T)





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import math

import numpy as np # linear algebra



def calc_wavelength(rel_perm, tesla):

    wavelength_air = 300e6 / (42.58e6 * tesla)

    if rel_perm == 0:

        return np.NaN

    wavelength_relative = wavelength_air / math.sqrt(rel_perm)

    

    return wavelength_relative

df_15T['Wavelength (m)'] = df_15T.apply(lambda row: calc_wavelength(row['Permittivity'], 1.5), axis=1)

df_30T['Wavelength (m)'] = df_30T.apply(lambda row: calc_wavelength(row['Permittivity'], 3.0), axis=1)

df_70T['Wavelength (m)'] = df_70T.apply(lambda row: calc_wavelength(row['Permittivity'], 7.0), axis=1)



# Print example (7.0T) dataframe:

print('Dielectrical properties at 7T')

print(df_70T)

print('1.5 Tesla')

df_15T.loc[df_15T['Tissue'].str.contains('Brain')]
print('3.0 Tesla')

df_30T.loc[df_30T['Tissue'].str.contains('Brain')]
print('7.0 Tesla')

df_70T.loc[df_70T['Tissue'].str.contains('Brain')]
import scipy.io

import numpy.ma as ma

import matplotlib.pyplot as plt



# Load matlab file

B1cal_matlab = scipy.io.loadmat('/kaggle/input/b1cal-3t/B1cal_3T.mat')

B1 = B1cal_matlab['B1']

mask = B1cal_matlab['mask']



# Determine number of channels

tx_channels = np.shape(B1)[-1]



# Mask the B1 maps and create sensitivity maps

mask_full = np.repeat(mask[..., np.newaxis], tx_channels, axis=2)

B1m = ma.masked_array(B1, mask=np.logical_not(mask_full))



# Determine global plotting window

vmin = np.percentile(np.abs(B1m), 1)

vmax = np.percentile(np.abs(B1m), 99)



# Create the grid figure

fig, axis = plt.subplots(2, tx_channels, sharex=True, sharey=True)

fig.set_size_inches(24, 4)



# Loop over channels

for chan in range(tx_channels):

    # Show B1 intensity

    im0 = axis[0, chan].imshow(np.abs(B1m[:, :, chan]), vmin=vmin, vmax=vmax)

    axis[0, chan].set_title('Chan #' + str(chan+1))

    # Show B1 phase with cyclic colormap

    im1 = axis[1, chan].imshow(np.angle(B1m[:, :, chan]), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')



# Label the axis

axis[0, 0].set_ylabel('B1 amplitude')

axis[1, 0].set_ylabel('B1 phase')



# Show the colorbar

fig.colorbar(im0, ax=axis[0, :], shrink=0.8)

fig.colorbar(im1, ax=axis[1, :], shrink=0.8)

fig.show()

# Sum over the channel dimension

B1m_sum = np.sum(B1m, axis=2)



# Determine global plotting window again

vmin = np.percentile(np.abs(B1m_sum), 1)

vmax = np.percentile(np.abs(B1m_sum), 99)



# Create new figure

fig, axis = plt.subplots(1, 2, sharey=True)

fig.set_size_inches(8, 4)



# Plot B1 amplitude

im0 = axis[0].imshow(np.abs(B1m_sum), vmin=vmin, vmax=vmax)

axis[0].set_title('B1 amplitude at 0°')



# Plot B1 phase

im1 = axis[1].imshow(np.angle(B1m_sum), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')

axis[1].set_title('B1 phase at 0°')



# Show the colorbar

fig.colorbar(im0, ax=axis[0], shrink=0.55)

fig.colorbar(im1, ax=axis[1], shrink=0.55)

fig.show()
# Create a copy of the matrix, we will overwrite the data but preserve the mask

B1m_CP = B1m.copy()



# Create CP phase offsets by spacing the channels evenly around a full rotation

CP_shim = np.linspace(0, 2*np.pi, tx_channels, endpoint=False)



# Apply the phase shift

for chan in range(tx_channels):

    B1m_CP[..., chan] = B1m[..., chan] * np.exp(1j * CP_shim[chan])



# Calculate the resulting total B1 field

B1m_CP_sum = np.sum(B1m_CP, axis=2)





# Plot shim settings per channels

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']

ax.bar(ch_names, CP_shim/np.pi*180)

ax.set_title('Channel offsets for center shim')

ax.set_ylabel('Angle (degrees)')



# Determine global plotting window again

vmin = np.percentile(np.abs(B1m_CP_sum), 1)

vmax = np.percentile(np.abs(B1m_CP_sum), 99)



# Create new figure

fig, axis = plt.subplots(1, 2, sharey=True)

fig.set_size_inches(8, 4)



# Plot B1 amplitude

im0 = axis[0].imshow(np.abs(B1m_CP_sum), vmin=vmin, vmax=vmax)

axis[0].set_title('B1 amplitude at CP')



# Plot B1 phase

im1 = axis[1].imshow(np.angle(B1m_CP_sum), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')

axis[1].set_title('B1 phase at CP')



# Show the colorbar

fig.colorbar(im0, ax=axis[0], shrink=0.55)

fig.colorbar(im1, ax=axis[1], shrink=0.55)

fig.show()
#import scipy.stats

#scipy.stats.variation



B1m_cv = np.abs(B1m_sum).std() / np.abs(B1m_sum).mean()

B1m_CP_cv = np.abs(B1m_CP_sum).std() / np.abs(B1m_CP_sum).mean()



print('CV 0deg: ' + str(round(B1m_cv, 2)))

print('CV CP: ' + str(round(B1m_CP_cv, 2)))
# Create a copy of the matrix, we will overwrite the data but preserve the mask

B1m_cshim = B1m.copy()



# Calc center coordinate

centerx = math.floor(np.shape(B1)[0] / 2)

centery = math.floor(np.shape(B1)[1] / 2)



# Create array to save shim

center_shim_degrees = np.ndarray(tx_channels)



# Apply phase shift to 0deg in center

for chan in range(tx_channels):

    center_shim_degrees[chan] = -np.angle(B1m[centerx, centery, chan]) / (2*np.pi) * 360

    B1m_cshim[..., chan] = B1m[..., chan] * np.exp(-1j * np.angle(B1m[centerx, centery, chan]))

    #print(np.angle(B1_center_shim[centerx, centery, chan]))

    

# Plot shim settings per channels

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']

ax.bar(ch_names, center_shim_degrees)

ax.set_title('Channel offsets for center shim')

ax.set_ylabel('Angle (degrees)')
# Calc resulting fields

B1m_cshim_sum = np.sum(B1m_cshim, axis=2)

    

# Determine global plotting window again

vmin = np.percentile(np.abs(B1m_cshim_sum), 1)

vmax = np.percentile(np.abs(B1m_cshim_sum), 99)



# Create new figure

fig, axis = plt.subplots(1, 2, sharey=True)

fig.set_size_inches(8, 4)



# Plot B1 amplitude

im0 = axis[0].imshow(np.abs(B1m_cshim_sum), vmin=vmin, vmax=vmax)

axis[0].set_title('B1 amplitude at center shim')



# Plot B1 phase

im1 = axis[1].imshow(np.angle(B1m_cshim_sum), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')

axis[1].set_title('B1 phase at center shim')



# Show the colorbar

fig.colorbar(im0, ax=axis[0], shrink=0.55)

fig.colorbar(im1, ax=axis[1], shrink=0.55)

fig.show()





# B1 strength in center

print('B1 center 0deg: ' + str(np.abs(B1m_sum[centerx, centery])))

print('B1 center CP: ' + str(np.abs(B1m_CP_sum[centerx, centery])))

print('B1 center cshim: ' + str(np.abs(B1m_cshim_sum[centerx, centery])))



# B1 strength in center, relative to 0deg

print('B1% center 0deg: 100.0%')

print('B1% center CP: ' + str( round(np.abs(B1m_CP_sum[centerx, centery])/np.abs(B1m_sum[centerx, centery])*100) ) + '%')

print('B1% center cshim: ' + str( round(np.abs(B1m_cshim_sum[centerx, centery])/np.abs(B1m_sum[centerx, centery])*100) ) + '%')



# Calc CV of new center shimset

B1m_cshim_cv = np.abs(B1m_cshim).std() / np.abs(B1m_cshim).mean()

print('CV center shim: ' + str(round(B1m_cshim_cv, 2)))
from scipy.optimize import minimize, least_squares



# Define the costfunction that will be minimized

def costfunc(shims, B1_masked, exp_goal = 2):

    # Apply shims

    B1sens = B1_masked.copy()

    for chan in range(len(shims)):

        B1sens[..., chan] = B1sens[..., chan] * np.exp(1j * shims[chan])

        

    # Calculate cost function

    B1shim = np.abs(np.sum(B1sens, axis=2))

    cost = B1shim.std() / (B1shim.mean()**exp_goal)



    return cost



# Create phase bounds

bounds = [(0, 2*np.pi) for i in range(8)]  



# Perform actual optimization of the RF fields

# Make an educated guess and start with CP shim

res = minimize(costfunc, CP_shim, args=(B1m), method='trust-constr', bounds=bounds, options={'disp': True, 'maxiter':1e3, 'xtol':1e-4})



# Center phases around 0

shim_avg = res.x.mean()

shim = res.x - shim_avg

# Plot shim settings per channels

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']

ax.bar(ch_names, shim/np.pi*180)

ax.set_title('Channel offsets for center shim')

ax.set_ylabel('Angle (degrees)')





# Create a copy of the matrix, we will overwrite the data but preserve the mask

B1m_shim = B1m.copy()



# Apply shim

for chan in range(tx_channels):

    B1m_shim[..., chan] = B1m[..., chan] * np.exp(1j * res.x[chan])



# Calc resulting fields

B1m_shim_sum = np.sum(B1m_shim, axis=2)





# Determine global plotting window again

vmin = np.percentile(np.abs(B1m_shim_sum), 1)

vmax = np.percentile(np.abs(B1m_shim_sum), 99)



# Create new figure

fig, axis = plt.subplots(1, 2, sharey=True)

fig.set_size_inches(8, 4)



# Plot B1 amplitude

im0 = axis[0].imshow(np.abs(B1m_shim_sum), vmin=vmin, vmax=vmax)

axis[0].set_title('B1 amplitude at optimised shim')



# Plot B1 phase

im1 = axis[1].imshow(np.angle(B1m_shim_sum), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')

axis[1].set_title('B1 phase at optimised shim')



# Show the colorbar

fig.colorbar(im0, ax=axis[0], shrink=0.55)

fig.colorbar(im1, ax=axis[1], shrink=0.55)

fig.show()





# B1 strength in center

print('B1 in center full shim: ' + str(np.abs(B1m_shim_sum[centerx, centery])))



# B1 strength in center, relative to 0deg

print('B1% in center full shim: ' + str( round(np.abs(B1m_shim_sum[centerx, centery])/np.abs(B1m_sum[centerx, centery])*100) ) + '%')



# Calc CV of new center shimset

B1_res_cv = B1m_shim_sum.std() / B1m_shim_sum.mean()

print('CV center shim: ' + str(round(B1_res_cv, 2)))
import seaborn as sns



# Create new figure

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



# Plot the full optimized shim

sns.distplot(np.abs(B1m_shim_sum).compressed(), hist=True, kde=True, 

             bins=50, color = 'darkblue', label='Optimized shim',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Plot the CP shim

sns.distplot(np.abs(B1m_CP_sum).compressed(), hist=True, kde=True, 

             bins=50, color = 'darkgreen', label='CP (quadrature)',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Plot the center voxel shim

sns.distplot(np.abs(B1m_cshim_sum).compressed(), hist=True, kde=True, 

             bins=50, color = 'darkred', label='Center shim',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Add some labels

ax.set_title('B1 density profile (and histogram) for different shims')

ax.set_ylabel('Amount of binned voxels')

ax.set_xlabel('B1 (au.)')



# Show the legend

plt.legend()

plt.show()
# Create new figure

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])





for exp_goal in np.arange(0.5,9.5,1):

    

    # Perform optimization of the RF fields, but now also provide the exp_goal

    res = minimize(costfunc, CP_shim, args=(B1m, exp_goal), method='trust-constr', bounds=bounds, options={'disp': True, 'maxiter':1e3, 'xtol':1e-4})



    # Calc resulting fields

    B1m_expshim = B1m.copy()

    for chan in range(tx_channels):

        B1m_expshim[..., chan] = B1m[..., chan] * np.exp(1j * res.x[chan])

    B1m_expshim_sum = np.sum(B1m_expshim, axis=2)



    # Plot the optimized shim

    sns.distplot(np.abs(B1m_expshim_sum).compressed(), hist=True, kde=True, 

                 bins=50, label='exp_goal= ' + str(exp_goal),

                 hist_kws={'edgecolor':'black'},

                 kde_kws={'linewidth': 4}, ax=ax)



# Add some labels

ax.set_title('B1 density profile (and histogram) for different shims')

ax.set_ylabel('Amount of binned voxels')

ax.set_xlabel('B1 (au.)')



# Show the legend

plt.legend()

plt.show()
# Define the MLS costfunction that will be minimized

def MLS_costfunc(shims, B1_masked, B1_target = 1e-6):

    # Apply shims

    B1sens = B1_masked.copy()

    for chan in range(len(shims)):

        B1sens[..., chan] = B1sens[..., chan] * np.exp(1j * shims[chan])

        

    # Calculate cost function

    B1shim = np.abs(np.sum(B1sens, axis=2))

    cost = ((B1shim-B1_target)**2).sum()



    return cost



# Perform actual optimization of the RF fields

# Make an educated guess and start with CP shim

res = minimize(MLS_costfunc, CP_shim, args=(B1m), method='trust-constr', bounds=bounds, options={'disp': True, 'maxiter':1e3, 'xtol':1e-20, 'gtol': 1e-20,})





# Center phases around 0

shim_avg = res.x.mean()

shim = res.x - shim_avg



# Plot shim settings per channels

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']

ax.bar(ch_names, shim/np.pi*180)

ax.set_title('Channel offsets for MLS shim')

ax.set_ylabel('Angle (degrees)')

plt.show()
# Calc resulting fields

B1m_MLSshim = B1m.copy()

for chan in range(tx_channels):

    B1m_MLSshim[..., chan] = B1m[..., chan] * np.exp(1j * res.x[chan])

B1m_MLSshim_sum = np.sum(B1m_MLSshim, axis=2)





# Create new figure

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



# Plot the MLS shim

sns.distplot(np.abs(B1m_MLSshim_sum).compressed(), hist=True, kde=True, 

             bins=50, label='MLS shim',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Plot the modified CV shim

sns.distplot(np.abs(B1m_shim_sum).compressed(), hist=True, kde=True, 

             bins=50, label='Modified CV shim',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Add some labels

ax.set_title('B1 density profile (and histogram) for different shims')

ax.set_ylabel('Amount of binned voxels')

ax.set_xlabel('B1 (au.)')



# Show the legend

plt.legend()

plt.show()
# Perform MLS optimization of the RF fields, with slightly raised B1_target

res = minimize(MLS_costfunc, CP_shim, args=(B1m, 1.05e-6), method='trust-constr', bounds=bounds, options={'disp': True, 'maxiter':1e3, 'xtol':1e-20, 'gtol': 1e-20,})



# Calc resulting fields

B1m_MLS2shim = B1m.copy()

for chan in range(tx_channels):

    B1m_MLS2shim[..., chan] = B1m[..., chan] * np.exp(1j * res.x[chan])

B1m_MLS2shim_sum = np.sum(B1m_MLS2shim, axis=2)



# Create new figure

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



# Plot the MLS shim

sns.distplot(np.abs(B1m_MLS2shim_sum).compressed(), hist=True, kde=True, 

             bins=50, label='MLS shim',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Plot the modified CV shim

sns.distplot(np.abs(B1m_shim_sum).compressed(), hist=True, kde=True, 

             bins=50, label='Modified CV shim',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax=ax)



# Add some labels

ax.set_title('B1 density profile (and histogram) for different shims')

ax.set_ylabel('Amount of binned voxels')

ax.set_xlabel('B1 (au.)')



# Show the legend

plt.legend()

plt.show()







# Determine global plotting window again

vmin = np.percentile(np.abs(B1m_shim_sum), 1)

vmax = np.percentile(np.abs(B1m_shim_sum), 99)



# Create new figure

fig, axis = plt.subplots(1, 2, sharey=True)

fig.set_size_inches(8, 4)



# Plot B1 amplitude MLS shim

im0 = axis[0].imshow(np.abs(B1m_MLS2shim_sum), vmin=vmin, vmax=vmax)

axis[0].set_title('B1 MLS shim')



# Plot B1 amplitude modified CV shim

im1 = axis[1].imshow(np.abs(B1m_shim_sum), vmin=vmin, vmax=vmax)

axis[1].set_title('B1 modified CV shim')



# Show the colorbar

fig.colorbar(im0, ax=axis[0], shrink=0.55)

fig.colorbar(im1, ax=axis[1], shrink=0.55)

fig.show()

# Create new figure

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])





for B1_target in np.arange(0.8e-6,1.25e-6,0.05e-6):

    

    # Perform MLS optimization of the RF fields, but now provide the B1_target

    res = minimize(MLS_costfunc, CP_shim, args=(B1m, B1_target), method='trust-constr', bounds=bounds, options={'disp': True, 'maxiter':1e3, 'xtol':1e-20, 'gtol': 1e-20,})



    # Calc resulting fields

    B1m_MLStshim = B1m.copy()

    for chan in range(tx_channels):

        B1m_MLStshim[..., chan] = B1m[..., chan] * np.exp(1j * res.x[chan])

    B1m_MLStshim_sum = np.sum(B1m_MLStshim, axis=2)





    # Plot the optimized shim

    sns.distplot(np.abs(B1m_MLStshim_sum).compressed(), hist=False, kde=True, 

                 bins=50, label=f'B1_target= {B1_target:.2E}',

                 hist_kws={'edgecolor':'black'},

                 kde_kws={'linewidth': 4}, ax=ax)



# Add some labels

ax.set_title('B1 density profile (and histogram) for different shims')

ax.set_ylabel('Amount of binned voxels')

ax.set_xlabel('B1 (au.)')



# Show the legend

plt.legend()

plt.show()