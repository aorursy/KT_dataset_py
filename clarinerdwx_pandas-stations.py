from ftplib import FTP

import matplotlib.pyplot as plt

import pandas as pd

import gzip

from io import BytesIO

import shutil

from datetime import datetime, timedelta

import os

import seaborn as sns

import numpy as np

from glob import glob



!pip install pandarallel

from pandarallel import pandarallel



pandarallel.initialize()

%matplotlib inline
isd_history = pd.read_csv("../input/noaa-global-surface-summary-of-the-day/isd-history.csv")
def get_ids(icao: str):

    icao = icao.upper()

    wban = isd_history.query(f"ICAO == '{icao}'").WBAN.to_list()[0]

    usaf = isd_history.query(f"ICAO == '{icao}'").USAF.to_list()[0]

    return wban, usaf
years = pd.date_range(start="2006", end="2019", freq="AS")

icao = "KASH"

(wban, usaf) = get_ids(icao)
def get_file(year):

    filename = f"{usaf}-{wban}-{year}"

    print(filename)

    with FTP("ftp.ncdc.noaa.gov") as ftp, BytesIO() as flo:

        ftp.login()

        ftp.retrbinary(f"RETR pub/data/noaa/isd-lite/{year}/{filename}.gz", flo.write)

        flo.seek(0)

        with open(f"{filename}.gz", "wb") as fout, gzip.GzipFile(fileobj = flo) as gzipobj:

            shutil.copyfileobj(gzipobj, fout)
for year in list(years.year):

    get_file(year)
filelist = glob("*.gz")

df_list = []

for file in filelist:

    with open(file, "r") as f:

        df_list.append(

           pd.read_csv(

               f,

               delim_whitespace=True,

               header=None,

               names=[

                   "year",

                   "month",

                   "day",

                   "hour",

                   "tmpc",

                   "dwpc",

                   "mslp",

                   "wdir",

                   "wspd",

                   "skct",

                   "pr1h",

                   "pr6h"

               ]

           )

        )

df = pd.concat(df_list)
df
df['Timestamp'] = df.parallel_apply(

    lambda row: datetime(row.year, row.month, row.day, row.hour) - timedelta(hours=6), axis=1)

df['Timestamp'] = pd.to_datetime(df.Timestamp)

df.index = pd.DatetimeIndex(df.Timestamp)

df = df.drop(columns='Timestamp')

df['tmpc'] /= 10.

df['dwpc'] /= 10.

df['wspd'] /= 10.

df['mslp'] /= 10.

df['pr1h'] /= 10.

df['pr6h'] /= 10.

df['doy'] = df.index.dayofyear

df['woy'] = df.index.weekofyear

# clean up missing data

df = df.mask(df.lt(-999.8),np.nan)
def max_grouper(var):

    grouped = df[f'{var}'].groupby([df.index.hour,df.woy]).max(numeric_only=True)

    return grouped.unstack(level = 0)

def min_grouper(var):

    grouped = df[f'{var}'].groupby([df.index.hour,df.woy]).min(numeric_only=True)

    return grouped.unstack(level = 0)

def mean_grouper(var):

    grouped = df[f'{var}'].groupby([df.index.hour,df.woy]).mean(numeric_only=True)

    return grouped.unstack(level = 0)
mean_dwpt = mean_grouper("dwpc")

mean_temp = mean_grouper("tmpc")

mean_dpdp = mean_temp - mean_dwpt
sns.set_style("darkgrid")

plt.figure(figsize=(9,9))

sns.heatmap(mean_dpdp, cmap="mako")

plt.title(f'Diurnal cycle of mean dewpoint depression at {icao}\nby week (2006-2020)')

plt.ylabel('Week of year')

plt.ylim([0,52])

plt.xlabel('Hour (LST)')

# plt.savefig(f'./figs/{icao}_dewpoint_depression_mean_diur.png',dpi=200)