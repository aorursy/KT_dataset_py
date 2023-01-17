# This cell sets things up. Run it without modifying!

# Load packages
import numpy as np
import scipy.io.netcdf as netcdf
import matplotlib.pyplot as plt
import copy
from polar_spectra_utility import *
# Run this cell without modifying anything

# .. We'll be using the polar spectrum and the Oklahoma spectrum again, so load them in now.
#    Polar spectrum
with netcdf.netcdf_file('../input/polar-spectra-data/smtaerich1nf1turnX1.c1.20101228.000313.cdf', 'r') as f:
    nu_polar = copy.deepcopy(f.variables['wnum'][:])
    radiance_polar = copy.deepcopy(np.mean(f.variables['mean_rad'][3300:3375, :], axis=0))

#    Oklahoma spectrum
filename_oklahoma  = '../input/polar-spectra-data/sgp_aeri_20170706.000443.txt' 
oklahoma_data = np.loadtxt(filename_oklahoma)
nu_oklahoma = oklahoma_data[:,0]
radiance_oklahoma = oklahoma_data[:,1]

# .. Constants that will be used
h = 6.62607004e-34    # m2 kg / s
c = 2.99792458e8      # m / s
k = 1.38064852e-23    # J/K


%matplotlib notebook
nu = np.linspace(100, 4e6, 1000)   # m-1

T = 6000
eterm = np.exp(h * c * nu / (k * T))
planck_function_6000K = 2 * h * c**2 * nu**3 / (eterm - 1)

# Plot the blackbody radiation spectrum
# It is typical to plot spectra as a function of wavenumbers
# in cm^-1 rather than m^-1, so we divide nu by 100 below.
plt.figure()
plt.plot(nu/100, planck_function_6000K, label = '6000 K')
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('Radiance')
plt.legend()
T = 300
eterm = np.exp(h * c * nu / (k * T))
planck_function_300K = 2 * h * c**2 * nu**3 / (eterm - 1)

plt.figure()
plt.plot(nu/100, planck_function_6000K, label = 'sun: 6000 K')
plt.plot(nu/100, planck_function_300K, label = 'near-surface atmosphere: 300 K')
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('Radiance')
plt.legend()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('wav')
ax1.set_ylabel('Energy density', color=color)
ax1.plot(nu/100, planck_function_300K, color=color, label='300 K')
ax1.tick_params(axis='y', labelcolor=color)
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('Radiance')
ax1.legend(loc='best')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Radiance', color=color)  # we already handled the x-label with ax1
ax2.plot(nu/100, planck_function_6000K, color=color, label = '6000 k')
ax2.tick_params(axis='y', labelcolor=color)
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('Radiance')
ax2.legend(bbox_to_anchor=(0,0,1,.9))


fig.tight_layout()  # otherwise the right y-label is slightly clipped


# Modify the following according to the instructions.
co2 = 400
h2o = 40000000
ch4 = 2
other = 1
dT = 0

# Run a function that gets the radiance for your model atmosphere (do not modify)
nu_my_atmosphere, radiance_my_atmosphere, my_legend = get_my_radiance(
    co2, h2o, ch4, other, dT)

# Plot the spectrum for your atmosphere together with the Oklahoma and polar spectra
# (no need to modify this)
fig = plt.figure()
plt.plot(nu_polar, radiance_polar, 'c')
plt.plot(nu_my_atmosphere, radiance_my_atmosphere, color='orange')
plt.legend(['Polar Winter', my_legend])


# Plot the spectrum from your model atmosphere and the polar winter.
fig = plt.figure()

plt.plot(nu_polar, radiance_polar, 'c', label = 'Polar Winter')
plt.plot(nu_my_atmosphere, radiance_my_atmosphere, color='orange', label = my_legend)
plt.plot(nu/100, 1e5*planck_function_300K, label = 'Planck function for 300 K')  # add the Planck function for 300 K here
plt.legend()

# Zoom in on the region of interest
plt.xlim([450, 1800])
T = 235
eterm = np.exp(h*c*nu/(k*T))
planck_function_my_temperature = 2*h*c**2*nu**3/(eterm-1)

plt.plot(nu/100, 1e5*planck_function_my_temperature, label = 'Plank function for ' +str(T) + ' K')
plt.legend()
# Plot the Oklahoma spectrum together with the polar spectrum
plt.figure()
plt.plot(nu_polar, radiance_polar, label = 'Polar winter')
plt.plot(nu_oklahoma, radiance_oklahoma, label = 'Oklahoma summer')

plt.xlim([490, 1800])          
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('Radiance (mW / [m$^2$ sr$^{-1}$ cm$^{-1}$])')


# Create and plot blackbody function below. 
T = 302
eterm = np.exp(h*c*nu/(k*T))
planck_function_my_temperature = 2*h*c**2*nu**3/(eterm-1)

plt.plot(nu/100, 1e5*planck_function_my_temperature, label = 'Planck function for ' +str(T) + ' K')
plt.legend()
