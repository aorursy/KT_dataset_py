# workaround to extract the zip files

!cat ../input/paper_details.csv.zip.part-* > /dev/shm/paper_details.csv.zip

!unzip '/dev/shm/paper_details.csv.zip' -d '/dev/shm/'

!ls '/dev/shm/'

!rm /dev/shm/paper_details.csv.zip
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from dask import dataframe  #import the datframe datastructure, behaves mostly like pandas

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')  # use a nicer default styling
paper_dk = dataframe.read_csv('/dev/shm/paper_details.csv',blocksize=4000000)  # point to csv

year_month_group = paper_dk.groupby(['year', 'month']).size()  # describe the computation to perform
ym_df = year_month_group.compute()  # actually read and compute the result.
plt.figure(figsize=(12,8))

ym_df.unstack().T.sum().plot()  # Plot publications by year

plt.title('Articles published per year')
ym_60_16 = ym_df.unstack().loc[1964:]  # take only the years for which there is monthly data
plt.figure(figsize=(12,8))

ym_60_16.boxplot()

plt.title('Boxplot of publication by month from 1964 to 2016')

plt.ylabel('Number of papers')

plt.xlabel('Month')

plt.tight_layout();