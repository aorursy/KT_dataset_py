import tifffile as tiff



import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data_path = '/kaggle/input/ds4g-environmental-insights-explorer'



image = '/eie_data/s5p_no2/s5p_no2_20190629T174803_20190705T194117.tif'



data = tiff.imread(data_path + image)



print('Data shape:', data.shape)
print('Data sample')

print(data[:2])
data[0][0]
f = plt.figure()

f.set_size_inches(12, 9)

for i in range(12):

    plt.subplot(3, 4, i+1)

    sns.heatmap(data[:, :, i], cbar=False)

    f

    ax = plt.gca()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
print('{:.0f}% of the dataset is null'.format(

    np.isnan(data[:, :, 0]).sum() / np.multiply(*data[:, :, 0].shape)*100

))



print('Last measurement at y index {}'.format(np.argwhere(np.isnan(data[0, :, 0])).min()-1))
data_nn = data[:, :177, :]  # no nulls
titles = ['NO2_column_number_density',

          'tropospheric_NO2_column_number_density', 

          'stratospheric_NO2_column_number_density',

          'NO2_slant_column_number_density']



f = plt.figure()

f.set_size_inches(8, 8)

for i in range(4):

    plt.subplot(2, 2, i+1)

    sns.heatmap(data_nn[:, :, i], cbar=False)

    f

    ax = plt.gca()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    plt.title(titles[i], fontsize=10)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
for i in range(4):

    print('{}: {:.2E} mol/m^2'.format(titles[i], np.nanmean(data_nn[:, :, i])))

titles = ['tropopause_pressure', 'absorbing_aerosol_index', 'cloud_fraction']

f = plt.figure()

f.set_size_inches(12,4)

for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.heatmap(data_nn[:, :, 4+i], cbar=False)

    f

    ax = plt.gca()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    plt.title(titles[i], fontsize=16)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
for i in range(3):

    print('{}: {:.2f}'.format(titles[i], np.nanmean(data_nn[:, :, i+4])))

titles = ['sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle',

          'solar_azimuth_angle', 'solar_zenith_angle']



f = plt.figure()

f.set_size_inches(12, 8)

for i in range(5):

    plt.subplot(2, 3, i+1)

    sns.heatmap(data_nn[:, :, 7+i], cbar=False)

    f

    ax = plt.gca()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    plt.title(titles[i], fontsize=16)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
for i in range(5):

    print('{}: {:.2f}'.format(titles[i], np.nanmean(data_nn[:, :, i+7])))