# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
regions = pd.read_csv(os.path.join(dirname, 'covid19_italy_region.csv' ))
regions.head()
regions['Date'] = pd.to_datetime(regions['Date'])
italy = regions.groupby('Date').sum()
italy.info()
# we will focus only on hospitalize patients, intensive care patients and deaths
# as Postive cases are not a relible feature
cols = ['HospitalizedPatients', 'IntensiveCarePatients', 'Deaths']

# need to add some columns with daily variations
daily_cols=[]
for c in cols:
    daily_cols.append('Daily_' + c)
italy_ds = italy[cols]
# calculate daily variation for 'HospitalizedPatients', 'IntensiveCarePatients' and 'Deaths'
for c in cols:
    italy_ds['Daily_' + c] = italy_ds[c].shift(-1) - italy_ds[c] 
italy_ds
plt.figure(figsize=(24,4))
i=1
for c in cols:
    plt.subplot(1,3,i)

    plt.plot(italy_ds[c])
    plt.title(c)
    plt.xlabel('day')
    plt.ylabel('Number of ' + str(c))
    plt.xticks(rotation=45)
    i=i+1

plt.figure(figsize=(24,4))
i=1
for c in daily_cols:
    plt.subplot(1,3,i)

    plt.plot(italy_ds[c])
    plt.title(c)
    plt.xlabel('day')
    plt.ylabel('Number of ' + str(c))
    i=i+1
plt.show()
# resample for 3days mean
plt.figure(figsize=(24,4))
i=1
for c in daily_cols:
    plt.subplot(1,3,i)
#     plt.figure(figsize=(8,4))

    plt.plot(italy_ds[c].resample('3D').mean())
    plt.title('variation_in_' + c)
    plt.xlabel('day')
    plt.ylabel('Number of ' + str(c))
    i=i+1
plt.show()
# It looks like from the eastern weekend pressure on the health care system is increasing after a huge dropping during the previous weeks, can it be the loosing of social distancing
# will lead to a new peak in the next weeks? We will see