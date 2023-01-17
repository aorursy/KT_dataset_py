# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/weekhack2/Train.csv')

test=pd.read_csv('/kaggle/input/weekhack2/Test.csv')

submit=pd.read_csv('/kaggle/input/weekhack2/SampleSubmission (1).csv')
train_original=train.copy()

test_original=test.copy()
train.shape,test.shape
test.isnull().sum()[0:40]
train.isnull().sum()[40:77]
train.columns
x=train.drop("target", axis=1)

y=train['target']
train.columns
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_absorbing_aerosol_index']);

plt.subplot(122) 

train['L3_NO2_absorbing_aerosol_index'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_absorbing_aerosol_index'].fillna(test['L3_NO2_absorbing_aerosol_index'].mean(), inplace=True)

train['L3_NO2_absorbing_aerosol_index'].fillna(train['L3_NO2_absorbing_aerosol_index'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_cloud_fraction']);

plt.subplot(122) 

train['L3_NO2_cloud_fraction'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_cloud_fraction'].fillna(test['L3_NO2_cloud_fraction'].median(), inplace=True)

train['L3_NO2_cloud_fraction'].fillna(train['L3_NO2_cloud_fraction'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_sensor_altitude']);

plt.subplot(122) 

train['L3_NO2_sensor_altitude'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_sensor_altitude'].fillna(test['L3_NO2_sensor_altitude'].median(), inplace=True)

train['L3_NO2_sensor_altitude'].fillna(train['L3_NO2_sensor_altitude'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_sensor_azimuth_angle']);

plt.subplot(122) 

train['L3_NO2_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_sensor_azimuth_angle'].fillna(test['L3_NO2_sensor_azimuth_angle'].mean(), inplace=True)

train['L3_NO2_sensor_azimuth_angle'].fillna(train['L3_NO2_sensor_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_solar_zenith_angle']);

plt.subplot(122) 

train['L3_NO2_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_solar_zenith_angle'].fillna(test['L3_NO2_solar_zenith_angle'].mean(), inplace=True)

train['L3_NO2_solar_zenith_angle'].fillna(train['L3_NO2_solar_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_NO2_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_sensor_zenith_angle'].fillna(test['L3_NO2_sensor_zenith_angle'].mean(), inplace=True)

train['L3_NO2_sensor_zenith_angle'].fillna(train['L3_NO2_sensor_altitude'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_solar_azimuth_angle']);

plt.subplot(122) 

train['L3_NO2_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_solar_azimuth_angle'].fillna(test['L3_NO2_solar_azimuth_angle'].median(), inplace=True)

train['L3_NO2_solar_azimuth_angle'].fillna(train['L3_NO2_solar_azimuth_angle'].median(), inplace=True)


#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_tropopause_pressure']);

plt.subplot(122) 

train['L3_NO2_tropopause_pressure'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_tropopause_pressure'].fillna(test['L3_NO2_tropopause_pressure'].mean(), inplace=True)

train['L3_NO2_tropopause_pressure'].fillna(train['L3_NO2_tropopause_pressure'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_O3_O3_effective_temperature']);

plt.subplot(122) 

train['L3_O3_O3_effective_temperature'].plot.box(figsize=(16,5))

plt.show()
test['L3_O3_O3_effective_temperature'].fillna(test['L3_O3_O3_effective_temperature'].median(), inplace=True)

train['L3_O3_O3_effective_temperature'].fillna(train['L3_O3_O3_effective_temperature'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_O3_sensor_azimuth_angle']);

plt.subplot(122) 

train['L3_O3_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_O3_sensor_azimuth_angle'].fillna(test['L3_O3_sensor_azimuth_angle'].mean(), inplace=True)

train['L3_O3_sensor_azimuth_angle'].fillna(train['L3_O3_sensor_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_O3_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_O3_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_O3_sensor_zenith_angle'].fillna(test['L3_O3_sensor_zenith_angle'].mean(), inplace=True)

train['L3_O3_sensor_zenith_angle'].fillna(train['L3_O3_sensor_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_O3_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_O3_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_O3_sensor_zenith_angle'].fillna(test['L3_O3_sensor_zenith_angle'].mean(), inplace=True)

train['L3_O3_sensor_zenith_angle'].fillna(train['L3_O3_sensor_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_O3_solar_azimuth_angle']);

plt.subplot(122) 

train['L3_O3_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_solar_azimuth_angle'].fillna(test['L3_NO2_solar_azimuth_angle'].median(), inplace=True)

train['L3_NO2_solar_azimuth_angle'].fillna(train['L3_NO2_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_O3_solar_zenith_angle']);

plt.subplot(122) 

train['L3_O3_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_O3_solar_zenith_angle'].fillna(test['L3_O3_solar_zenith_angle'].mean(), inplace=True)

train['L3_O3_solar_zenith_angle'].fillna(train['L3_O3_solar_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CO_cloud_height']);

plt.subplot(122) 

train['L3_CO_cloud_height'].plot.box(figsize=(16,5))

plt.show()
test['L3_CO_cloud_height'].fillna(test['L3_CO_cloud_height'].median(), inplace=True)

train['L3_CO_cloud_height'].fillna(train['L3_CO_cloud_height'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CO_sensor_altitude']);

plt.subplot(122) 

train['L3_CO_sensor_altitude'].plot.box(figsize=(16,5))

plt.show()
test['L3_CO_sensor_altitude'].fillna(test['L3_CO_sensor_altitude'].median(), inplace=True)

train['L3_CO_sensor_altitude'].fillna(train['L3_CO_sensor_altitude'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CO_sensor_azimuth_angle']);

plt.subplot(122) 

train['L3_CO_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CO_sensor_azimuth_angle'].fillna(test['L3_CO_sensor_azimuth_angle'].mean(), inplace=True)

train['L3_CO_sensor_azimuth_angle'].fillna(train['L3_CO_sensor_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CO_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_CO_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CO_sensor_zenith_angle'].fillna(test['L3_NO2_solar_azimuth_angle'].mean(), inplace=True)

train['L3_CO_sensor_zenith_angle'].fillna(train['L3_NO2_solar_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CO_solar_azimuth_angle']);

plt.subplot(122) 

train['L3_CO_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CO_solar_azimuth_angle'].fillna(test['L3_CO_solar_azimuth_angle'].median(), inplace=True)

train['L3_CO_solar_azimuth_angle'].fillna(train['L3_CO_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CO_solar_zenith_angle']);

plt.subplot(122) 

train['L3_CO_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CO_solar_zenith_angle'].fillna(test['L3_CO_solar_zenith_angle'].mean(), inplace=True)

train['L3_CO_solar_zenith_angle'].fillna(train['L3_CO_solar_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_cloud_fraction']);

plt.subplot(122) 

train['L3_HCHO_cloud_fraction'].plot.box(figsize=(16,5))

plt.show()
test['L3_HCHO_cloud_fraction'].fillna(test['L3_HCHO_cloud_fraction'].median(), inplace=True)

train['L3_HCHO_cloud_fraction'].fillna(train['L3_HCHO_cloud_fraction'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_sensor_azimuth_angle']);

plt.subplot(122) 

train['L3_HCHO_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_HCHO_sensor_azimuth_angle'].fillna(test['L3_HCHO_sensor_azimuth_angle'].mean(), inplace=True)

train['L3_HCHO_sensor_azimuth_angle'].fillna(train['L3_HCHO_sensor_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_HCHO_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_HCHO_sensor_zenith_angle'].fillna(test['L3_HCHO_sensor_zenith_angle'].mean(), inplace=True)

train['L3_HCHO_sensor_zenith_angle'].fillna(train['L3_HCHO_sensor_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_solar_azimuth_angle']);

plt.subplot(122) 

train['L3_HCHO_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_HCHO_solar_azimuth_angle'].fillna(test['L3_HCHO_solar_azimuth_angle'].median(), inplace=True)

train['L3_HCHO_solar_azimuth_angle'].fillna(train['L3_HCHO_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_solar_zenith_angle']);

plt.subplot(122) 

train['L3_HCHO_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_HCHO_solar_zenith_angle'].fillna(test['L3_HCHO_solar_zenith_angle'].mean(), inplace=True)

train['L3_HCHO_solar_zenith_angle'].fillna(train['L3_HCHO_solar_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_tropospheric_HCHO_column_number_density_amf']);

plt.subplot(122) 

train['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].plot.box(figsize=(16,5))

plt.show()
test['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].fillna(test['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].mean(), inplace=True)

train['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].fillna(train['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_cloud_base_height']);

plt.subplot(122) 

train['L3_CLOUD_cloud_base_height'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_cloud_base_height'].fillna(test['L3_CLOUD_cloud_base_height'].median(), inplace=True)

train['L3_CLOUD_cloud_base_height'].fillna(train['L3_CLOUD_cloud_base_height'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_cloud_base_pressure']);

plt.subplot(122) 

train['L3_CLOUD_cloud_base_pressure'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_cloud_base_pressure'].fillna(test['L3_CLOUD_cloud_base_pressure'].median(), inplace=True)

train['L3_CLOUD_cloud_base_pressure'].fillna(train['L3_CLOUD_cloud_base_pressure'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_cloud_fraction']);

plt.subplot(122) 

train['L3_CLOUD_cloud_fraction'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_cloud_fraction'].fillna(test['L3_CLOUD_cloud_fraction'].mean(), inplace=True)

train['L3_CLOUD_cloud_fraction'].fillna(train['L3_CLOUD_cloud_fraction'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_cloud_optical_depth']);

plt.subplot(122) 

train['L3_CLOUD_cloud_optical_depth'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_cloud_optical_depth'].fillna(test['L3_CLOUD_cloud_optical_depth'].median(), inplace=True)

train['L3_CLOUD_cloud_optical_depth'].fillna(train['L3_CLOUD_cloud_optical_depth'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_cloud_top_height']);

plt.subplot(122) 

train['L3_CLOUD_cloud_top_height'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_cloud_top_height'].fillna(test['L3_CLOUD_cloud_top_height'].median(), inplace=True)

train['L3_CLOUD_cloud_top_height'].fillna(train['L3_CLOUD_cloud_top_height'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_cloud_top_pressure']);

plt.subplot(122) 

train['L3_CLOUD_cloud_top_pressure'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_cloud_top_pressure'].fillna(test['L3_CLOUD_cloud_top_pressure'].mean(), inplace=True)

train['L3_CLOUD_cloud_top_pressure'].fillna(train['L3_CLOUD_cloud_top_pressure'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_sensor_azimuth_angle']);

plt.subplot(122) 

train['L3_CLOUD_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_sensor_azimuth_angle'].fillna(test['L3_CLOUD_sensor_azimuth_angle'].mean(), inplace=True)

train['L3_CLOUD_sensor_azimuth_angle'].fillna(train['L3_CLOUD_sensor_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_CLOUD_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
#test['L3_CLOUD_sensor_zenith_angle'].fillna(test['L3_CLOUD_sensor_zenith_angle'].mean(), inplace=True)

#train['L3_CLOUD_sensor_zenith_angle'].fillna(train['L3_CLOUD_sensor_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_solar_azimuth_angle']);

plt.subplot(122) 

train['L3_CLOUD_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_solar_azimuth_angle'].fillna(test['L3_CLOUD_solar_azimuth_angle'].median(), inplace=True)

train['L3_CLOUD_solar_azimuth_angle'].fillna(train['L3_CLOUD_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_solar_zenith_angle']);

plt.subplot(122) 

train['L3_CLOUD_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
#test['L3_CLOUD_solar_zenith_angle'].fillna(test['L3_CLOUD_solar_zenith_angle'].mean(), inplace=True)

#train['L3_CLOUD_solar_zenith_angle'].fillna(train['L3_CLOUD_solar_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_CLOUD_surface_albedo']);

plt.subplot(122) 

train['L3_CLOUD_surface_albedo'].plot.box(figsize=(16,5))

plt.show()
test['L3_CLOUD_surface_albedo'].fillna(test['L3_CLOUD_surface_albedo'].median(), inplace=True)

train['L3_CLOUD_surface_albedo'].fillna(train['L3_NO2_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_AER_AI_absorbing_aerosol_index']);

plt.subplot(122) 

train['L3_AER_AI_absorbing_aerosol_index'].plot.box(figsize=(16,5))

plt.show()
test['L3_AER_AI_absorbing_aerosol_index'].fillna(test['L3_AER_AI_absorbing_aerosol_index'].mean(), inplace=True)

train['L3_AER_AI_absorbing_aerosol_index'].fillna(train['L3_AER_AI_absorbing_aerosol_index'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_AER_AI_sensor_altitude']);

plt.subplot(122) 

train['L3_AER_AI_sensor_altitude'].plot.box(figsize=(16,5))

plt.show()
test['L3_AER_AI_sensor_altitude'].fillna(test['L3_AER_AI_sensor_altitude'].median(), inplace=True)

train['L3_AER_AI_sensor_altitude'].fillna(train['L3_AER_AI_sensor_altitude'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_AER_AI_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_AER_AI_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_AER_AI_sensor_zenith_angle'].fillna(test['L3_AER_AI_sensor_zenith_angle'].mean(), inplace=True)

train['L3_AER_AI_sensor_zenith_angle'].fillna(train['L3_AER_AI_sensor_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_AER_AI_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_AER_AI_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_AER_AI_sensor_zenith_angle'].fillna(test['L3_AER_AI_sensor_zenith_angle'].mean(), inplace=True)

train['L3_AER_AI_sensor_zenith_angle'].fillna(train['L3_AER_AI_sensor_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train[ 'L3_AER_AI_solar_azimuth_angle']);

plt.subplot(122) 

train[ 'L3_AER_AI_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_AER_AI_solar_zenith_angle'].fillna(test['L3_AER_AI_solar_azimuth_angle'].median(), inplace=True)

train['L3_AER_AI_solar_zenith_angle'].fillna(train['L3_AER_AI_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_AER_AI_solar_zenith_angle']);

plt.subplot(122) 

train['L3_AER_AI_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_AER_AI_solar_zenith_angle'].fillna(test['L3_AER_AI_solar_zenith_angle'].median(), inplace=True)

train['L3_AER_AI_solar_zenith_angle'].fillna(train['L3_AER_AI_solar_zenith_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_SO2_column_number_density_amf']);

plt.subplot(122) 

train['L3_SO2_SO2_column_number_density_amf'].plot.box(figsize=(16,5))

plt.show()
test['L3_SO2_SO2_column_number_density_amf'].fillna(test['L3_SO2_SO2_column_number_density_amf'].median(), inplace=True)

train['L3_SO2_SO2_column_number_density_amf'].fillna(train['L3_SO2_SO2_column_number_density_amf'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_absorbing_aerosol_index']);

plt.subplot(122) 

train['L3_SO2_absorbing_aerosol_index'].plot.box(figsize=(16,5))

plt.show()
test['L3_SO2_absorbing_aerosol_index'].fillna(test['L3_SO2_absorbing_aerosol_index'].mean(), inplace=True)

train['L3_SO2_absorbing_aerosol_index'].fillna(train['L3_SO2_absorbing_aerosol_index'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_cloud_fraction']);

plt.subplot(122) 

train['L3_SO2_cloud_fraction'].plot.box(figsize=(16,5))

plt.show()
test['L3_SO2_cloud_fraction'].fillna(test['L3_SO2_cloud_fraction'].median(), inplace=True)

train['L3_SO2_cloud_fraction'].fillna(train['L3_SO2_cloud_fraction'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_sensor_azimuth_angle']);

plt.subplot(122) 

train['L3_SO2_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_NO2_solar_azimuth_angle'].fillna(test['L3_NO2_solar_azimuth_angle'].median(), inplace=True)

train['L3_NO2_solar_azimuth_angle'].fillna(train['L3_NO2_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_sensor_zenith_angle']);

plt.subplot(122) 

train['L3_SO2_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_SO2_sensor_azimuth_angle'].fillna(test['L3_SO2_sensor_azimuth_angle'].mean(), inplace=True)

train['L3_SO2_sensor_azimuth_angle'].fillna(train['L3_SO2_sensor_azimuth_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_solar_azimuth_angle']);

plt.subplot(122) 

train['L3_SO2_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_SO2_solar_azimuth_angle'].fillna(test['L3_SO2_solar_azimuth_angle'].median(), inplace=True)

train['L3_SO2_solar_azimuth_angle'].fillna(train['L3_SO2_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_SO2_solar_zenith_angle']);

plt.subplot(122) 

train['L3_SO2_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_SO2_solar_zenith_angle'].fillna(test['L3_SO2_solar_zenith_angle'].mean(), inplace=True)

train['L3_SO2_solar_zenith_angle'].fillna(train['L3_SO2_solar_zenith_angle'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_tropospheric_NO2_column_number_density']);

plt.subplot(122) 

train['L3_NO2_tropospheric_NO2_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_HCHO_tropospheric_HCHO_column_number_density']);

plt.subplot(122) 

train['L3_HCHO_tropospheric_HCHO_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(train['L3_NO2_NO2_column_number_density']);

plt.subplot(122) 

train['L3_NO2_NO2_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_NO2_NO2_slant_column_number_density']);

plt.subplot(122) 

test['L3_NO2_NO2_slant_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_NO2_stratospheric_NO2_column_number_density']);

plt.subplot(122) 

test['L3_NO2_stratospheric_NO2_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#train['L3_NO2_stratospheric_NO2_column_number_density'].replace(train.L3_NO2_stratospheric_NO2_column_number_density>7,0,inplace=True)

#test['L3_NO2_stratospheric_NO2_column_number_density'].replace(test.L3_NO2_stratospheric_NO2_column_number_density>7,0,inplace=True)

#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_NO2_tropospheric_NO2_column_number_density']);

plt.subplot(122) 

test['L3_NO2_tropospheric_NO2_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#train['L3_NO2_tropospheric_NO2_column_number_density'].replace(train.L3_NO2_tropospheric_NO2_column_number_density>0.0008,0,inplace=True)

#test['L3_NO2_tropospheric_NO2_column_number_density'].replace(test.L3_NO2_tropospheric_NO2_column_number_density>0.0008,0,inplace=True)

#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CO_CO_column_number_density']);

plt.subplot(122) 

test['L3_CO_CO_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#train['L3_CO_CO_column_number_density'].replace(train.L3_HCHO_HCHO_slant_column_number_density>0.14,0,inplace=True)

#test['L3_CO_CO_column_number_density'].replace(test.L3_HCHO_HCHO_slant_column_number_density>0.14,0,inplace=True)

#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CO_H2O_column_number_density']);

plt.subplot(122) 

test['L3_CO_H2O_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_HCHO_HCHO_slant_column_number_density']);

plt.subplot(122) 

test['L3_HCHO_HCHO_slant_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#train['L3_HCHO_HCHO_slant_column_number_density'].replace(train.L3_HCHO_HCHO_slant_column_number_density>0.0004,0,inplace=True)

#train['L3_HCHO_HCHO_slant_column_number_density'].replace(train.L3_HCHO_HCHO_slant_column_number_density<-0.0004,0,inplace=True)

#test['L3_HCHO_HCHO_slant_column_number_density'].replace(test.L3_HCHO_HCHO_slant_column_number_density>0.0004,0,inplace=True)

#test['L3_HCHO_HCHO_slant_column_number_density'].replace(test.L3_HCHO_HCHO_slant_column_number_density<-0.0004,0,inplace=True)
#data = data.query('L3_HCHO_tropospheric_HCHO_column_number_density >-0.0004 and L3_HCHO_tropospheric_HCHO_column_number_density < 0.000')
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_HCHO_tropospheric_HCHO_column_number_density']);

plt.subplot(122) 

test['L3_HCHO_tropospheric_HCHO_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#train['L3_HCHO_tropospheric_HCHO_column_number_density'].replace(train.L3_HCHO_tropospheric_HCHO_column_number_density>0.0009,0,inplace=True)

#train['L3_HCHO_tropospheric_HCHO_column_number_density'].replace(train.L3_HCHO_tropospheric_HCHO_column_number_density<-0.0006,0,inplace=True)

#test['L3_HCHO_tropospheric_HCHO_column_number_density'].replace(test.L3_HCHO_tropospheric_HCHO_column_number_density>0.0009,0,inplace=True)

#test['L3_HCHO_tropospheric_HCHO_column_number_density'].replace(test.L3_HCHO_tropospheric_HCHO_column_number_density<-0.0006,0,inplace=True)
#data = data.query('L3_HCHO_tropospheric_HCHO_column_number_density >-0.0006 and L3_HCHO_tropospheric_HCHO_column_number_density < 0.0008')
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_SO2_SO2_column_number_density']);

plt.subplot(122) 

test['L3_SO2_SO2_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#train['L3_SO2_SO2_column_number_density'].replace(train.L3_SO2_SO2_column_number_density>0.02,0,inplace=True)

#train['L3_SO2_SO2_column_number_density'].replace(train.L3_SO2_SO2_column_number_density<-0.02,0,inplace=True)

#test['L3_SO2_SO2_column_number_density'].replace(test.L3_SO2_SO2_column_number_density>0.02,0,inplace=True)

#train['L3_SO2_SO2_column_number_density'].replace(test.L3_SO2_SO2_column_number_density<-0.02,0,inplace=True)
#data = data.query('L3_SO2_SO2_slant_column_number_density >-0.02 and L3_SO2_SO2_slant_column_number_density < 0.02')
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_SO2_SO2_slant_column_number_density']);

plt.subplot(122) 

test['L3_SO2_SO2_slant_column_number_density'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_CH4_column_volume_mixing_ratio_dry_air']);

plt.subplot(122) 

test['L3_CH4_CH4_column_volume_mixing_ratio_dry_air'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_CH4_column_volume_mixing_ratio_dry_air'].fillna(test['L3_CH4_CH4_column_volume_mixing_ratio_dry_air'].mean(), inplace=True)

train['L3_CH4_CH4_column_volume_mixing_ratio_dry_air'].fillna(train['L3_CH4_CH4_column_volume_mixing_ratio_dry_air'].mean(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_aerosol_height']);

plt.subplot(122) 

test['L3_CH4_aerosol_height'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_aerosol_height'].fillna(test['L3_CH4_aerosol_height'].median(), inplace=True)

train['L3_CH4_aerosol_height'].fillna(train['L3_CH4_aerosol_height'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_aerosol_optical_depth']);

plt.subplot(122) 

test['L3_CH4_aerosol_optical_depth'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_aerosol_optical_depth'].fillna(test['L3_CH4_aerosol_optical_depth'].median(), inplace=True)

train['L3_CH4_aerosol_optical_depth'].fillna(train['L3_CH4_aerosol_optical_depth'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_sensor_azimuth_angle']);

plt.subplot(122) 

test['L3_CH4_sensor_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_sensor_azimuth_angle'].fillna(test['L3_CH4_sensor_azimuth_angle'].median(), inplace=True)

train['L3_CH4_sensor_azimuth_angle'].fillna(train['L3_CH4_sensor_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_sensor_zenith_angle']);

plt.subplot(122) 

test['L3_CH4_sensor_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_sensor_zenith_angle'].fillna(test['L3_CH4_sensor_zenith_angle'].median(), inplace=True)

train['L3_CH4_sensor_zenith_angle'].fillna(train['L3_CH4_sensor_zenith_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_solar_azimuth_angle']);

plt.subplot(122) 

test['L3_CH4_solar_azimuth_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_solar_azimuth_angle'].fillna(test['L3_CH4_solar_azimuth_angle'].median(), inplace=True)

train['L3_CH4_solar_azimuth_angle'].fillna(train['L3_CH4_solar_azimuth_angle'].median(), inplace=True)
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(test['L3_CH4_solar_zenith_angle']);

plt.subplot(122) 

test['L3_CH4_solar_zenith_angle'].plot.box(figsize=(16,5))

plt.show()
test['L3_CH4_solar_zenith_angle'].fillna(test['L3_CH4_solar_zenith_angle'].mean(), inplace=True)

train['L3_CH4_solar_zenith_angle'].fillna(train['L3_CH4_solar_zenith_angle'].mean(), inplace=True)
#train['L3_SO2_SO2_slant_column_number_density'].replace(train.L3_SO2_SO2_slant_column_number_density>0.002,0,inplace=True)

#train['L3_SO2_SO2_slant_column_number_density'].replace(train.L3_SO2_SO2_slant_column_number_density<-0.001,0,inplace=True)

#test['L3_SO2_SO2_slant_column_number_density'].replace(test.L3_SO2_SO2_slant_column_number_density>0.002,0,inplace=True)

#train['L3_SO2_SO2_slant_column_number_density'].replace(test.L3_SO2_SO2_slant_column_number_density<-0.001,0,inplace=True)
train['L3_NO2_NO2_column_number_density'].fillna(train['L3_NO2_NO2_column_number_density'].median(), inplace=True)

train['L3_NO2_NO2_slant_column_number_density'].fillna(train['L3_NO2_stratospheric_NO2_column_number_density'].median(), inplace=True)

train['L3_NO2_tropospheric_NO2_column_number_density'].fillna(train['L3_NO2_tropospheric_NO2_column_number_density'].mean(), inplace=True)

train['L3_CO_CO_column_number_density'].fillna(train['L3_CO_CO_column_number_density'].median(), inplace=True)

train['L3_CO_H2O_column_number_density'].fillna(train['L3_CO_H2O_column_number_density'].median(), inplace=True)

train['L3_HCHO_HCHO_slant_column_number_density'].fillna(train['L3_HCHO_HCHO_slant_column_number_density'].mean(), inplace=True)

train['L3_HCHO_tropospheric_HCHO_column_number_density'].fillna(train['L3_HCHO_tropospheric_HCHO_column_number_density'].mean(), inplace=True)

train['L3_SO2_SO2_column_number_density'].fillna(train['L3_SO2_SO2_column_number_density'].mean(), inplace=True)

train['L3_SO2_SO2_slant_column_number_density'].fillna(train['L3_SO2_SO2_slant_column_number_density'].mean(), inplace=True)

train['L3_NO2_tropospheric_NO2_column_number_density'].fillna(train['L3_NO2_tropospheric_NO2_column_number_density'].mean(), inplace=True)

train['L3_HCHO_tropospheric_HCHO_column_number_density'].fillna(train['L3_HCHO_tropospheric_HCHO_column_number_density'].mean(), inplace=True)
test['L3_NO2_NO2_column_number_density'].fillna(test['L3_NO2_NO2_column_number_density'].median(), inplace=True)

test['L3_NO2_NO2_slant_column_number_density'].fillna(test['L3_NO2_stratospheric_NO2_column_number_density'].median(), inplace=True)

test['L3_NO2_tropospheric_NO2_column_number_density'].fillna(test['L3_NO2_tropospheric_NO2_column_number_density'].mean(), inplace=True)

test['L3_CO_CO_column_number_density'].fillna(test['L3_CO_CO_column_number_density'].median(), inplace=True)

test['L3_CO_H2O_column_number_density'].fillna(test['L3_CO_H2O_column_number_density'].median(), inplace=True)

test['L3_HCHO_HCHO_slant_column_number_density'].fillna(test['L3_HCHO_HCHO_slant_column_number_density'].mean(), inplace=True)

test['L3_HCHO_tropospheric_HCHO_column_number_density'].fillna(test['L3_HCHO_tropospheric_HCHO_column_number_density'].mean(), inplace=True)

test['L3_SO2_SO2_column_number_density'].fillna(test['L3_SO2_SO2_column_number_density'].mean(), inplace=True)

test['L3_SO2_SO2_slant_column_number_density'].fillna(test['L3_SO2_SO2_slant_column_number_density'].mean(), inplace=True)

test['L3_NO2_tropospheric_NO2_column_number_density'].fillna(test['L3_NO2_tropospheric_NO2_column_number_density'].mean(), inplace=True)

test['L3_HCHO_tropospheric_HCHO_column_number_density'].fillna(test['L3_HCHO_tropospheric_HCHO_column_number_density'].mean(), inplace=True)
z = train.target



features =[

       'L3_NO2_NO2_column_number_density',

       'L3_NO2_NO2_slant_column_number_density',

       'L3_NO2_absorbing_aerosol_index', 'L3_NO2_cloud_fraction',

       'L3_NO2_sensor_altitude', 'L3_NO2_sensor_azimuth_angle',

       'L3_NO2_sensor_zenith_angle', 'L3_NO2_solar_azimuth_angle',

       'L3_NO2_solar_zenith_angle',

       'L3_NO2_stratospheric_NO2_column_number_density',

       'L3_NO2_tropopause_pressure',

       'L3_NO2_tropospheric_NO2_column_number_density',

       'L3_O3_O3_column_number_density', 'L3_O3_O3_effective_temperature',

       'L3_O3_cloud_fraction', 'L3_O3_sensor_azimuth_angle',

       'L3_O3_sensor_zenith_angle', 'L3_O3_solar_azimuth_angle',

       'L3_O3_solar_zenith_angle', 'L3_CO_CO_column_number_density',

       'L3_CO_H2O_column_number_density', 'L3_CO_cloud_height',

       'L3_CO_sensor_altitude', 'L3_CO_sensor_azimuth_angle',

       'L3_CO_sensor_zenith_angle', 'L3_CO_solar_azimuth_angle',

       'L3_CO_solar_zenith_angle', 'L3_HCHO_HCHO_slant_column_number_density',

       'L3_HCHO_cloud_fraction', 'L3_HCHO_sensor_azimuth_angle',

       'L3_HCHO_sensor_zenith_angle', 'L3_HCHO_solar_azimuth_angle',

       'L3_HCHO_solar_zenith_angle',

       'L3_HCHO_tropospheric_HCHO_column_number_density',

       'L3_HCHO_tropospheric_HCHO_column_number_density_amf',

       'L3_CLOUD_cloud_base_height', 'L3_CLOUD_cloud_base_pressure',

       'L3_CLOUD_cloud_fraction', 'L3_CLOUD_cloud_optical_depth',

       'L3_CLOUD_cloud_top_height', 'L3_CLOUD_cloud_top_pressure',

       'L3_CLOUD_sensor_azimuth_angle', 'L3_CLOUD_sensor_zenith_angle',

       'L3_CLOUD_solar_azimuth_angle', 'L3_CLOUD_solar_zenith_angle',

       'L3_CLOUD_surface_albedo', 'L3_AER_AI_absorbing_aerosol_index',

       'L3_AER_AI_sensor_altitude', 'L3_AER_AI_sensor_azimuth_angle',

       'L3_AER_AI_sensor_zenith_angle', 'L3_AER_AI_solar_azimuth_angle',

       'L3_AER_AI_solar_zenith_angle', 'L3_SO2_SO2_column_number_density',

       'L3_SO2_SO2_column_number_density_amf',

       'L3_SO2_SO2_slant_column_number_density',

       'L3_SO2_absorbing_aerosol_index', 'L3_SO2_cloud_fraction',

       'L3_SO2_sensor_azimuth_angle', 'L3_SO2_sensor_zenith_angle',

       'L3_SO2_solar_azimuth_angle', 'L3_SO2_solar_zenith_angle',

       'L3_CH4_CH4_column_volume_mixing_ratio_dry_air',

       'L3_CH4_aerosol_height', 'L3_CH4_aerosol_optical_depth',

       'L3_CH4_sensor_azimuth_angle', 'L3_CH4_sensor_zenith_angle',

       'L3_CH4_solar_azimuth_angle', 'L3_CH4_solar_zenith_angle']

X = train[features].copy()

X_test = test[features].copy()



from sklearn.model_selection import train_test_split

# Break off validation set from training data

X_train, X_valid, z_train, z_valid = train_test_split(X, z, train_size=0.5280623097817194, test_size=0.4719376902182806,

                                                      random_state=100)
from sklearn.ensemble import RandomForestRegressor

my_model =RandomForestRegressor(n_estimators=200, random_state=400)



from sklearn.impute import SimpleImputer



my_impute = SimpleImputer()

imputed_X_train = my_impute.fit_transform(X)

imputed_X_test1 = my_impute.transform(X_test)



#imputed_X_train

#imputed_X_test

#y_train

#y_test





my_model.fit(imputed_X_train, z)



preds_test = my_model.predict(imputed_X_test1)

print(preds_test)



output = pd.DataFrame({'Place_ID X Date':submit['Place_ID X Date'],

                       'target':preds_test})

output.to_csv('SampleSubmission (1).csv', index=False)