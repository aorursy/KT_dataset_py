# Load packages for processing 



import math                  # basic math operations

import numpy as np           # data matrix manipulation

import scipy.io as sio       # matlab file reads

import matplotlib as plt     # plots

import matplotlib.cm as cm

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.io import netcdf as nc      # read netcdf files for UTIG data load



# Sets the size of all inline plots

plt.rcParams['figure.figsize'] = [20, 12]



# Load physical constants and site locations

c = 3e8                           # Speed of light in a vaccuum 

n_ice = math.sqrt(3.17)           # index of refraction for glacial ice



# Reference trace locations for the radargrams 

lat_rough = 75.1525

lon_rough = -56.202423

lat_smooth = 77.121733

lon_smooth = -50.458014



# Load raw radar data example

tmp1 = sio.loadmat('../input/greenland-mcords3/RawRadarExample.mat', struct_as_record = False, squeeze_me = True)

raw = tmp1['data'][3000:7000:1,::]



# Load the pulse compressed data examples

tmp2 = sio.loadmat('../input/greenland-mcords3/rds_20140426_01_041.mat', struct_as_record = False, squeeze_me = True)

pc_rough = tmp2['seg']

tmp3 = sio.loadmat('../input/greenland-mcords3/rds_20140508_01_061.mat', struct_as_record = False, squeeze_me = True)

pc_smooth = tmp3['seg'] 



# Load the SAR processed data examples

tmp4 = sio.loadmat('../input/greenland-mcords3/sar_20140426_01_041.mat', struct_as_record = False, squeeze_me = True)

sar_rough = tmp4['seg']

tmp5 = sio.loadmat('../input/greenland-mcords3/sar_20140508_01_061.mat', struct_as_record = False, squeeze_me = True)

sar_smooth = tmp5['seg']



%reset_selective -f tmp # clear temporary variables from memory
# Sets the size of the inline plot

plt.rcParams['figure.figsize'] = [20,12]



# Plot raw radargram



fig1, ax1 = plt.subplots()

raw_im = ax1.imshow(raw, cmap = plt.cm.Blues, aspect = 'auto', vmin = -1000, vmax = 1000)

ax1.set_title("Raw Radar Data", fontsize = 24, fontweight = 'bold')

ax1.set_xlabel("Trace Number", fontsize = 18, fontweight = 'bold')

ax1.set_ylabel("Fast Time Sample Number", fontsize = 18, fontweight = 'bold')
# Plot a single trace from the raw radargram 



plt.plot(raw[:,1000])

plt.xlabel('Fast Time Sample Number', fontsize = 18, fontweight = 'bold')

plt.ylabel('Amplitude', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Raw Radar Data')

plt.show()
# Plot the rough topography pulse compressed data



depth_rough = 0.5*(pc_rough.Time - pc_rough.Time[150])*(c/n_ice)

box = np.array([np.amin(pc_rough.Along_Track)/1000, np.amax(pc_rough.Along_Track)/1000, depth_rough[1000], depth_rough[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(pc_rough.Data[0:1000:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -180, vmax = -40, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Pulse Compressed Data - Rough Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot a representative trace from the rough topography pulse compressed data



ind_rough_pc = np.argmin(np.abs(pc_rough.Latitude - lat_rough + pc_rough.Longitude - lon_rough))



plt.plot(depth_rough, np.asarray(10*np.log10(np.square(np.absolute(pc_rough.Data[:,ind_rough_pc])))))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Pulse Compressed Data - Rough Topography', fontsize = 24, fontweight = 'bold')

plt.show()

# Plot the smooth topography pulse compressed data



depth_smooth = 0.5*(pc_smooth.Time - pc_smooth.Time[98])*(c/n_ice)

box = np.array([np.amin(pc_smooth.Along_Track)/1000, np.amax(pc_smooth.Along_Track)/1000, depth_smooth[1080], depth_smooth[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(pc_smooth.Data[0:1080:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -200, vmax = -50, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Pulse Compressed Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot a representative trace from the smooth topography pulse compressed data



ind_smooth_pc = np.argmin(np.abs(pc_smooth.Latitude - lat_smooth + pc_smooth.Longitude - lon_smooth))



plt.plot(depth_smooth, np.asarray(10*np.log10(np.square(np.absolute(pc_smooth.Data[:,ind_smooth_pc])))))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Pulse Compressed Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Implements coherent summation 



window = 20



width_rough = int(np.floor(pc_rough.Data.shape[1]/window))

uf_rough = {}

uf_rough["Data"] = np.empty((pc_rough.Data.shape[0], width_rough), dtype = np.complex64)

uf_rough["Latitude"] = np.empty(width_rough)

uf_rough["Longitude"] = np.empty(width_rough)

uf_rough["Along_Track"] = np.empty(width_rough)

uf_rough["Time"] = pc_rough.Time



for k in range(width_rough):

    uf_rough["Data"][:,k] = np.sum(pc_rough.Data[:,k*window:(k+1)*window:1],axis=1)

    uf_rough["Latitude"][k] = np.mean(pc_rough.Latitude[k*window:(k+1)*window:1])

    uf_rough["Longitude"][k] = np.mean(pc_rough.Longitude[k*window:(k+1)*window:1])

    uf_rough["Along_Track"][k] = np.mean(pc_rough.Along_Track[k*window:(k+1)*window:1])



width_smooth = int(np.floor(pc_smooth.Data.shape[1]/window))

uf_smooth = {}

uf_smooth["Data"] = np.empty((pc_smooth.Data.shape[0], width_smooth), dtype = np.complex64)

uf_smooth["Latitude"] = np.empty(width_smooth)

uf_smooth["Longitude"] = np.empty(width_smooth)

uf_smooth["Along_Track"] = np.empty(width_smooth)

uf_smooth["Time"] = pc_smooth.Time



for k in range(width_smooth):

    uf_smooth["Data"][:,k] = np.sum(pc_smooth.Data[:,k*window:(k+1)*window:1],axis=1)

    uf_smooth["Latitude"][k] = np.mean(pc_smooth.Latitude[k*window:(k+1)*window:1])

    uf_smooth["Longitude"][k] = np.mean(pc_smooth.Longitude[k*window:(k+1)*window:1])

    uf_smooth["Along_Track"][k] = np.mean(pc_smooth.Along_Track[k*window:(k+1)*window:1])
# Plot the rough topography unfocused data



depth_rough_uf = 0.5*(uf_rough["Time"] - uf_rough["Time"][150])*(c/n_ice)

box = np.array([np.amin(uf_rough["Along_Track"])/1000, np.amax(uf_rough["Along_Track"])/1000, depth_rough_uf[1000], depth_rough_uf[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(uf_rough["Data"][0:1000:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -150, vmax = -30, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Unfocused Data - Rough Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot the smooth topography unfocused data



depth_smooth_uf = 0.5*(uf_smooth["Time"] - uf_smooth["Time"][98])*(c/n_ice)

box = np.array([np.amin(uf_smooth["Along_Track"])/1000, np.amax(uf_smooth["Along_Track"])/1000, depth_smooth_uf[1080], depth_smooth_uf[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(uf_smooth["Data"][0:1080:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -150, vmax = -30, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Unfocused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot a representative trace from the rough topography unfocused data



ind_rough_uf = np.argmin(np.abs(uf_rough["Latitude"]- lat_rough + uf_rough["Longitude"] - lon_rough))



plt.plot(depth_rough_uf, np.asarray(10*np.log10(np.square(np.absolute(uf_rough["Data"][:,ind_rough_uf])))))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Unfocused Data - Rough Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Plot a representative trace from the smooth topography unfocused data



ind_smooth_uf = np.argmin(np.abs(uf_smooth["Latitude"]- lat_smooth + uf_smooth["Longitude"] - lon_smooth))



plt.plot(depth_smooth_uf, np.asarray(10*np.log10(np.square(np.absolute(uf_smooth["Data"][:,ind_smooth_uf])))))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Unfocused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Implement Multi-Looking



window = 10



width_rough = int(np.floor(uf_rough["Data"].shape[1]/window))

ml_rough = {}

ml_rough["Data"] = np.empty((uf_rough["Data"].shape[0], width_rough))

ml_rough["Latitude"] = np.empty(width_rough)

ml_rough["Longitude"] = np.empty(width_rough)

ml_rough["Along_Track"] = np.empty(width_rough)

ml_rough["Time"] = uf_rough["Time"]



for k in range(width_rough):

    ml_rough["Data"][:,k] = np.sum(np.square(np.abs(uf_rough["Data"][:,k*window:(k+1)*window:1])),axis=1)

    ml_rough["Latitude"][k] = np.mean(uf_rough["Latitude"][k*window:(k+1)*window:1])

    ml_rough["Longitude"][k] = np.mean(uf_rough["Longitude"][k*window:(k+1)*window:1])

    ml_rough["Along_Track"][k] = np.mean(uf_rough["Along_Track"][k*window:(k+1)*window:1])



width_smooth = int(np.floor(uf_smooth["Data"].shape[1]/window))

ml_smooth = {}

ml_smooth["Data"] = np.empty((uf_smooth["Data"].shape[0], width_smooth))

ml_smooth["Latitude"] = np.empty(width_smooth)

ml_smooth["Longitude"] = np.empty(width_smooth)

ml_smooth["Along_Track"] = np.empty(width_smooth)

ml_smooth["Time"] = uf_smooth["Time"]



for k in range(width_smooth):

    ml_smooth["Data"][:,k] = np.sum(np.square(np.abs(uf_smooth["Data"][:,k*window:(k+1)*window:1])),axis=1)

    ml_smooth["Latitude"][k] = np.mean(uf_smooth["Latitude"][k*window:(k+1)*window:1])

    ml_smooth["Longitude"][k] = np.mean(uf_smooth["Longitude"][k*window:(k+1)*window:1])

    ml_smooth["Along_Track"][k] = np.mean(uf_smooth["Along_Track"][k*window:(k+1)*window:1])
# Plot the rough topography multi-looked data



depth_rough_ml = 0.5*(ml_rough["Time"] - ml_rough["Time"][150])*(c/n_ice)

box = np.array([np.amin(ml_rough["Along_Track"])/1000, np.amax(ml_rough["Along_Track"])/1000, depth_rough_ml[1000], depth_rough_ml[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(ml_rough["Data"][0:1000:1,::]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -140, vmax = -10, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Multi-Looked Data - Rough Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot the smooth topography multi-looked data



depth_smooth_ml = 0.5*(ml_smooth["Time"] - ml_smooth["Time"][98])*(c/n_ice)

box = np.array([np.amin(ml_smooth["Along_Track"])/1000, np.amax(ml_smooth["Along_Track"])/1000, depth_smooth_ml[1080], depth_smooth_ml[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(ml_smooth["Data"][0:1080:1,::]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -140, vmax = -10, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Multi-Looked Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot a representative trace from the rough topography multi-looked data



ind_rough_ml = np.argmin(np.abs(ml_rough["Latitude"]- lat_rough + ml_rough["Longitude"] - lon_rough))



plt.plot(depth_rough_ml, np.asarray(10*np.log10(ml_rough["Data"][:,ind_rough_ml])))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Multi-Looked Data - Rough Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Plot a representative trace from the smooth topography multi-looked data



ind_smooth_ml = np.argmin(np.abs(ml_smooth["Latitude"]- lat_smooth + ml_smooth["Longitude"] - lon_smooth))



plt.plot(depth_smooth_ml, np.asarray(10*np.log10(ml_smooth["Data"][:,ind_smooth_ml])))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Multi-Looked Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Plot the rough topography SAR focused data



rough_left = np.argmin(np.abs(sar_rough.Latitude - pc_rough.Latitude[0] + sar_rough.Longitude - pc_rough.Longitude[0]))

rough_right = np.argmin(np.abs(sar_rough.Latitude - pc_rough.Latitude[-1] + sar_rough.Longitude - pc_rough.Longitude[-1]))



depth_rough_sar = 0.5*(sar_rough.Time - sar_rough.Time[101])*(c/n_ice)

box = np.array([np.amin(sar_rough.Along_Track[rough_left:rough_right])/1000, np.amax(sar_rough.Along_Track[rough_left:rough_right])/1000, depth_rough_sar[1000], depth_rough_sar[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(sar_rough.Data[0:1000:1,rough_left:rough_right]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -180, vmax = -40, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Focused Data - Rough Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot the smooth topography SAR focused data



smooth_left = np.argmin(np.abs(sar_smooth.Latitude - pc_smooth.Latitude[0] + sar_smooth.Longitude - pc_smooth.Longitude[0]))

smooth_right = np.argmin(np.abs(sar_smooth.Latitude - pc_smooth.Latitude[-1] + sar_smooth.Longitude - pc_smooth.Longitude[-1]))



depth_smooth_sar = 0.5*(sar_smooth.Time - sar_smooth.Time[102])*(c/n_ice)

box = np.array([np.amin(sar_smooth.Along_Track[smooth_left:smooth_right])/1000, np.amax(sar_smooth.Along_Track[smooth_left:smooth_right])/1000, depth_smooth_sar[1080], depth_smooth_sar[0]])



fig2, ax2 = plt.subplots()

pcr_im = ax2.imshow(10*np.log10(sar_smooth.Data[0:1080:1,smooth_left:smooth_right]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -180, vmax = -40, \

                    extent = box)

ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')

ax2.set_title('Focused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

fig2.colorbar(pcr_im, ax = ax2)
# Plot a representative trace from the rough topography multi-looked data



ind_rough_sar = np.argmin(np.abs(sar_rough.Latitude- lat_rough + sar_rough.Longitude - lon_rough))



plt.plot(depth_rough_sar, np.asarray(10*np.log10(sar_rough.Data[:,ind_rough_sar])))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Focused Data - Rough Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Plot a representative trace from the smooth topography SAR focused data



ind_smooth_sar = np.argmin(np.abs(sar_smooth.Latitude- lat_smooth + sar_smooth.Longitude - lon_smooth))



plt.plot(depth_smooth_sar, np.asarray(10*np.log10(sar_smooth.Data[:,ind_smooth_sar])))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Focused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')

plt.show()
# Load and plot CReSIS data for system comparison



cresis = sio.loadmat('../input/system-comparison/CReSIS_20131127_01_029.mat', struct_as_record = False, squeeze_me = True)



lat = -75.090606

lon = 123.297632



ilat = np.argmin(np.abs(cresis["Latitude"] - lat + cresis["Longitude"] - lon))

d = 0.5*(cresis["Time"] - cresis["Time"][75])*(c/n_ice)

box = np.array([0, cresis["Data"].shape[1], d[1500], d[0]])



fig, ax = plt.subplots()

cresis_im = ax.imshow(10*np.log10(cresis["Data"][0:1500:1,::]), cmap = 'Blues', aspect = 'auto', vmin = -170, vmax = -40, extent = box)

ax.set_xlabel('Trace Number', fontsize = 18, fontweight = 'bold')

ax.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax.set_title('Dome C CReSIS Data', fontsize = 24, fontweight = 'bold')

fig.colorbar(cresis_im, ax = ax)
# Plot a single trace for CReSIS



plt.plot(d, np.asarray(10*np.log10(cresis["Data"][:,1000])))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Dome C CReSIS Data', fontsize = 24, fontweight = 'bold')

plt.show()
# Load and plot UTIG data for system comparison



utig = nc.netcdf_file('../input/system-comparison/UTIG_IR1HI1B.nc', 'r')

latitude = np.asarray(utig.variables["lat"][:].copy())

longitude = np.asarray(utig.variables["lon"][:].copy())

time = np.asarray(utig.variables["fasttime"][:].copy())

power_high = np.transpose(np.asarray(utig.variables["amplitude_high_gain"][:,:].copy()))

power_low = np.transpose(np.asarray(utig.variables["amplitude_low_gain"][:,:].copy()))

utig.close()



ilat2 = np.argmin(np.abs(latitude - lat + longitude - lon))

d2 = 0.5*(time - time[170])*(1e-6)*(c/n_ice)

box = np.array([0, power_high.shape[1], d2[2550], d2[0]])



fig, ax = plt.subplots()

cresis_im = ax.imshow(power_high[0:2550:1,::], cmap = 'Blues', aspect = 'auto', vmin = 40, vmax = 140, extent = box)

ax.set_xlabel('Trace Number', fontsize = 18, fontweight = 'bold')

ax.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax.set_title('Dome C UTIG Data', fontsize = 24, fontweight = 'bold')

fig.colorbar(cresis_im, ax = ax)
# Plot a single trace for UTIG



plt.plot(d2, power_high[:,1000])

plt.plot(d2, power_low[:,1000] + 40)

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Dome C UTIG Data', fontsize = 24, fontweight = 'bold')

plt.show()
# Load and plot BAS data for system comparison



bas = sio.loadmat('../input/system-comparison/BAS_SectionW36.mat', struct_as_record = False, squeeze_me = True)



lat = -75.090606

lon = 123.297632



ilat3 = np.argmin(np.abs(bas["lat"] - lat + bas["lon"] - lon))

Fs = 22e6

del_z = 0.5*(1/Fs)*(c/n_ice)

d3 = np.multiply(del_z, range(bas["abit"].shape[0])) - 114.805

box = np.array([0, bas["abit"].shape[1], d3[-1], d3[0]])



fig, ax = plt.subplots()

cresis_im = ax.imshow(10*np.log10(bas["abit"]), cmap = 'Blues', aspect = 'auto', vmin = 0, vmax = 85, extent = box)

ax.set_xlabel('Trace Number', fontsize = 18, fontweight = 'bold')

ax.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

ax.set_title('Dome C BAS Data', fontsize = 24, fontweight = 'bold')

fig.colorbar(cresis_im, ax = ax)
# Plot a single trace for BAS



plt.plot(d3, 10*np.log10(bas["abit"][:,500]))

plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')

plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')

plt.title('Single Trace - Dome C BAS Data', fontsize = 24, fontweight = 'bold')

plt.show()