# linear algebra

import numpy as np 

# data processing, CSV file I/O ...

import pandas as pd 
# list files

! ls ../input/list-of-isro-satellites
# read data

df = pd.read_csv('../input/list-of-isro-satellites/raw.csv')

df.head()
df.info()
df.describe(include='all')
launch_mass_in_kg = df['Launch Mass'].str.extract('([0-9,.]*)')[0]

launch_mass_in_kg = launch_mass_in_kg.str.replace(',', '')

launch_mass_in_kg = pd.to_numeric(launch_mass_in_kg, errors='coerce')



df['Launch Mass'] = launch_mass_in_kg

df = df.rename(columns={"Launch Mass": "Launch Mass in kg"})
# dry mass in kg



dry_mass_in_kg = df['Dry Mass'].str.extract('([0-9,.]*)')[0]

dry_mass_in_kg = dry_mass_in_kg.str.replace(',', '')

dry_mass_in_kg = pd.to_numeric(dry_mass_in_kg, errors='coerce')



df['Dry Mass'] = dry_mass_in_kg

df = df.rename(columns={"Dry Mass": "Dry Mass in kg"})
on_board_power_in_W = df['On-board Power'].str.extract('(\d*)')[0]

on_board_power_in_W = pd.to_numeric(on_board_power_in_W, errors='coerce')



df['On-board Power'] = on_board_power_in_W

df = df.rename(columns={"On-board Power": "On-board Power in W"})
launch_date = df['Launch Date'].str.extract('(\d+ [A-Za-z]+ \d{4})')[0]

launch_date = pd.to_datetime(launch_date, errors='coerce')



launch_time = df['Launch Date'].str.extract('(\d{1,2}\:\d{1,2}(\:\d{1,2})?)')[0]



launch_time_zone = df['Launch Date'].str.extract('([A-Z]{3})')[0]



df['Launch Date'] = launch_date

df['Launch Time'] = launch_time

df['Launch Timezone'] = launch_time_zone
periapsis_in_km = df['Periapsis'].str.extract('([0-9,.]*)')[0].str.replace(',', '')

periapsis_in_km = pd.to_numeric(periapsis_in_km, errors='coerce')



apoapsis_in_km = df['Apoapsis'].str.extract('([0-9,.]*)')[0].str.replace(',', '')

apoapsis_in_km = pd.to_numeric(apoapsis_in_km, errors='coerce')



semi_major_axis_in_km = df['Semi-Major Axis'].str.extract('([0-9,.]*)')[0].str.replace(',', '')

semi_major_axis_in_km = pd.to_numeric(semi_major_axis_in_km, errors='coerce')



df['Periapsis'] = periapsis_in_km

df['Apoapsis'] = apoapsis_in_km

df['Semi-Major Axis'] = semi_major_axis_in_km



df = df.rename(columns={"Periapsis": "Periapsis in km", 

                        "Apoapsis": "Apoapsis in km", 

                        "Semi-Major Axis": "Semi-Major Axis in km"})
period = df['Period'].str.replace(' (mins|minutes).*', '')

period = period.str.replace(',', '')

period = pd.to_numeric(period, errors='coerce')

df['Period'] = period
inclinations_in_degrees = df['Inclination'].str.replace('[^0-9.]', '')

inclinations_in_degrees = pd.to_numeric(inclinations_in_degrees, errors='coerce')

df['Inclination'] = inclinations_in_degrees
epoch_date = df['Epoch Start'].str.extract('(\d+ [A-Za-z]+ \d{4})')[0]

epoch_date = pd.to_datetime(launch_date, errors='coerce')



epoch_time = df['Epoch Start'].str.extract('(\d{1,2}\:\d{1,2}(\:\d{1,2})?)')[0]



epoch_time_zone = df['Epoch Start'].str.extract('([A-Z]{3})')[0]



df['Epoch Start Date'] = epoch_date

df['Epoch Start Time'] = epoch_time

df['Epoch Start Timezone'] = epoch_time_zone
decay_date = pd.to_datetime(df['Decay Date'], errors='coerce')

df['Decay Date'] = decay_date
launch_vehicle_type = df['Launch Vehicle'].str.split('-| ').str[0]

df['Launch Vehicle Type'] = launch_vehicle_type

launch_vehicle_type.value_counts()
pd.DataFrame(launch_vehicle_type.value_counts()).plot(kind='bar')