# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gzip



# convert to human-readable temperatures, i.e., *C (:P)

def FtoC (Ftemp):

    return (Ftemp - 32.) * 5.0/9.0

# extract fields of interest ("STN", "YEAR", "MODA", "MIN", "TEMP", "MAX") from a single .op.gz line

def process_opgz_line (line):

    return [line[:5], line[14:18], line[18:22], line[110:116], line[24:30], line[102:108]]



# process a single .op.gz file

def process_opgz (opgz_path, moda=None, verbose=True):

    # read in the data

    station_data = []

    with gzip.open(opgz_path,'rb') as station_file:

        station_contents = station_file.readlines()[1:]

    columns = ["STN", "YEAR", "MODA", "MIN", "TEMP", "MAX"] # header

    # let's extract the data from their character-wise position (seems safest, ref. readme.txt)

    station_data += list(map(lambda line : [line[:5], line[14:18], line[18:22], line[110:116], line[24:30], line[102:108]], station_contents))

    if verbose: print(station_data[:5])

    station_df = pd.DataFrame(station_data, columns=columns)

    # cast to the right type

    for col in columns:

        station_df[col] = pd.to_numeric(station_df[col], errors='coerce')

    # select only the data for the current month and day number

    if moda != None:

        station_df.drop(station_df[station_df['MODA'] != moda].index, inplace=True)

    for col in ['MIN', 'TEMP', 'MAX']:

        # convert to human-readable temperatures, i.e., *C (:P)

        station_df[col] = station_df[col].apply(FtoC)

        # handle missing data

        for col in ['MIN', 'TEMP', 'MAX']:

            mask = (station_df[col] < -100.) | (station_df[col] > 100.)

            station_df[col].mask(mask, inplace=True)

    if verbose:

        print(station_df.head())

        print(station_df.describe())

    return station_df



# process a single year, i.e., .tar file

import tarfile

def process_tar (tar_path, moda=None, verbose=True):

    if verbose: print("Processing year data from file %s.." % tar_path)

    # extract the tarfile

    if verbose: print(' - extracting tarfile.. ', end='', flush=True)

    with tarfile.open(tar_path) as tar:

        tar.extractall(path='./temp')

    if verbose: print('done.', flush=True)

    # process all the op.gz files

    if verbose: print(" - processing .op.gz files.. ", flush=True)

    year_df = pd.DataFrame(columns=["STN", "YEAR", "MODA", "MIN", "TEMP", "MAX"])

    station_files = sorted(os.listdir("./temp"))

    for station_file in station_files:

        station_df = process_opgz("./temp/"+station_file, moda=moda, verbose=False)

        if station_df.shape[0] > 0:

            year_df = year_df.append(station_df)

    if verbose: print('    done.', flush=True)

    if verbose: print(' - averaging.. ', end='', flush=True)

    # average all data per day per station

    year_df = year_df.groupby(['YEAR', 'MODA', 'STN'], as_index=False).mean().reset_index()

    # average per day across all stations

    year_df = year_df.groupby(['YEAR', 'MODA'], as_index=False).mean().reset_index()

    year_df.drop(labels=['STN',], inplace=True, axis=1)

    if verbose: print('done.', flush=True)

    if verbose: print(year_df.head())

    if verbose: print(" - removing temporary files.. ", end='', flush=True)

    ! rm -r './temp'

    if verbose: print('done.', flush=True)

    if verbose: print(' - year %s done.' % tar_path, flush=True)

    return year_df
# find the last moda of the dataset

import os

year_files = sorted(os.listdir("../input/gsod_all_years/"))

df = process_tar("../input/gsod_all_years/" + year_files[-1], moda=None, verbose=False)

today = df['MODA'].max()

# since we've already loaded the last year, let's use it

df.drop(df[df['MODA'] != today].index, inplace=True)



# now process all the data; sit back and enjoy :D

from tqdm import tqdm

year_files = sorted(os.listdir("../input/gsod_all_years/"))

print("Processing data for the last 50 years..", flush=True)

for year_file in tqdm(year_files[-50:-1]):

    df = df.append(process_tar("../input/gsod_all_years/" + year_file, moda=today, verbose=False))

df.sort_values('YEAR', inplace=True)



# save the processed data for reuse

import pickle as pkl

with open('processed_data.pkl', 'wb') as f:

    pkl.dump(df, f)
# finally, let's plot the data

# based on Rachael's kernel (Thanks! :): https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-python?utm_medium=email&utm_source=intercom&utm_campaign=dashboarding-event



# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Scatter(x=df['YEAR'], y=df['MIN'], line=dict(color='black', dash='dash'), name='Min'),

        go.Scatter(x=df['YEAR'], y=df['TEMP'], marker=dict(color='red'), name='Avg', mode='markers'),

        go.Scatter(x=df['YEAR'], y=df['MAX'], line=dict(color='black', dash='dash'), name='Max')]



# make the date pretty

import datetime

dt = datetime.datetime.today()

dt = dt.replace(month=int(today/100))

dt = dt.replace(day=np.mod(today,100))



# specify the layout of our figure

layout = dict(title = 'Station-average surface temperature on %s' % dt.strftime("%b %d"),

              xaxis= dict(title= 'Year',zeroline= False),

              yaxis= dict(title= 'Temperature [C]',zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)