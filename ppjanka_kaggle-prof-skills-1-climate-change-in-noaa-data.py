# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# See how much data there is...
year_files = sorted(os.listdir("../input/gsod_all_years/"))
print(year_files[:5])
print(year_files[-5:])
print(len(year_files))
# First, let's import and look at a single dataset
import tarfile
pathstem = "../input/gsod_all_years/"
with tarfile.open(pathstem + year_files[-1]) as tar:
    i = 0
    for tarinfo in tar:
        #Credits: https://docs.python.org/3/library/tarfile.html
        print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")
        if tarinfo.isreg():
            print("a regular file.")
        elif tarinfo.isdir():
            print("a directory.")
        else:
            print("something else.")
        i += 1
        if i > 5: break # just show a couple of files
    tar.extractall(path='./temp')
station_files = sorted(os.listdir("./temp"))
print(station_files[-5:])
import gzip
with gzip.open("./temp/" + station_files[0],'rb') as station_file:
    i = 0
    for line in station_file:
        print(line)
        i+=1
        if i > 4: break
    station_df = pd.read_csv(station_file, sep=r'\s+')
print(station_df.head())
print(station_df.describe())
print(station_df.columns.values)

# NOTE that the header is missing the COUNT fields (ref. readme.txt), we'll need to add them manually
from datetime import datetime
today = datetime.today()
today = today.month*100+today.day
print(today)
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
df = process_opgz("./temp/" + station_files[0])
print(df.shape[0])
# process a single year, i.e., .tar file
import tarfile
from tqdm import tqdm
def process_tar (tar_path, moda=None, verbose=True):
    print("Processing year data from file %s.." % tar_path)
    # extract the tarfile
    print(' - extracting tarfile.. ', end='', flush=True)
    with tarfile.open(tar_path) as tar:
        if verbose:
            i = 0
            for tarinfo in tar:
                #Credits: https://docs.python.org/3/library/tarfile.html
                print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")
                if tarinfo.isreg():
                    print("a regular file.")
                elif tarinfo.isdir():
                    print("a directory.")
                else:
                    print("something else.")
                i += 1
                if i > 5: break # just show a couple of files
        tar.extractall(path='./temp')
    print('done.', flush=True)
    # process all the op.gz files
    print(" - processing .op.gz files.. ", flush=True)
    year_df = pd.DataFrame(columns=["STN", "YEAR", "MODA", "MIN", "TEMP", "MAX"])
    station_files = sorted(os.listdir("./temp"))
    for station_file in tqdm(station_files):
        station_df = process_opgz("./temp/"+station_file, moda=moda, verbose=False)
        if station_df.shape[0] > 0:
            year_df = year_df.append(station_df)
    print('    done.', flush=True)
    print(' - averaging.. ', end='', flush=True)
    # average all data per day per station
    year_df = year_df.groupby(['YEAR', 'MODA', 'STN'], as_index=False).mean().reset_index()
    # average per day across all stations
    year_df = year_df.groupby(['YEAR', 'MODA'], as_index=False).mean().reset_index()
    year_df.drop(labels=['STN',], inplace=True, axis=1)
    print('done.', flush=True)
    if verbose: print(year_df.head())
    print(" - removing temporary files.. ", end='', flush=True)
    ! rm -r './temp'
    print('done.', flush=True)
    print(' - year %s done.' % tar_path, flush=True)
    return year_df
# find the last moda of the dataset
year_files = sorted(os.listdir("../input/gsod_all_years/"))
df = process_tar("../input/gsod_all_years/" + year_files[-1], moda=None, verbose=False)
print(df.head())
print(df.describe())
today = df['MODA'].max()
print(today)
# now process all the data; sit back and enjoy :D
year_files = sorted(os.listdir("../input/gsod_all_years/"))
df = pd.DataFrame(columns=["YEAR", "MODA", "MIN", "TEMP", "MAX"])
# in principle, we can do the entire century, but, to keep it short,
# let's limit ourselves to the last 10 years (should take about an hour to process)
#  - it's, frankly, not very informative :/, but this is to keep this kernel able to run in an hour or so...
for year_file in year_files[-10:]:
    df = df.append(process_tar("../input/gsod_all_years/" + year_file, moda=today, verbose=False))
df.sort('YEAR', inplace=True)
# save the processed data for reuse
import pickle as pkl
with open('processed_data.pkl', 'wb') as f:
    pkl.dump(df, f)
! ls .
# finally, let's plot the data
print(df.head())
print(df.describe())

import matplotlib.pyplot as plt
plt.clf()
plt.scatter(df['YEAR'], df['TEMP'], color='red', label='Avg')
plt.plot(df['YEAR'], df['MIN'], 'k--', label='Min')
plt.plot(df['YEAR'], df['MAX'], 'k--', label='Max')
plt.grid()
plt.title('Station-average surface temperature on %i' % today)
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()
