# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
regions = pd.read_csv("/kaggle/input/covid19-in-italy/covid19_italy_region.csv")

regions['Date'] = pd.to_datetime(regions['Date']).dt.strftime('%m/%d')

#print(regions.columns)

#print(regions.describe)

#print(regions.groupby('Date').TotalPositiveCases.sum())

sns.set_style("whitegrid")


plt.figure(figsize=(17,6))

sns.lineplot(data= regions.groupby('Date').NewPositiveCases.sum(), label = "New Positive Cases").set_title('New positive cases')



plt.figure(figsize=(17,6))

sns.lineplot(data= regions.groupby('Date').TotalPositiveCases.sum() / regions.groupby('Date').TestsPerformed.sum(), label = "Positive cases over Tests Performed").set_title('Positive Tests ratio')
plt.figure(figsize=(17,6))

sns.lineplot(data= regions.groupby('Date').IntensiveCarePatients.sum() / regions.groupby('Date').HospitalizedPatients.sum(), label = "Intensive Patients over Hospitalized Patients").set_title('Overview Treatment of Confirmed Cases')

sns.lineplot(data= regions.groupby('Date').HomeConfinement.sum() / regions.groupby('Date').TotalPositiveCases.sum(), label = "Home confinement Cases over Total Positive Cases")

sns.lineplot(data= regions.groupby('Date').HospitalizedPatients.sum() / regions.groupby('Date').TotalPositiveCases.sum(), label = "Hospitalized Patients over Total Positive Cases")
plt.figure(figsize=(17,6))

#print(regions.groupby(['RegionName']).TotalPositiveCases.sum().sort_values(ascending=False))

regionName = regions.groupby(['RegionName']).TotalPositiveCases.sum().sort_values(ascending=False).index.tolist()

for region in range(1,6):

    sns.lineplot(y= regions.loc[regions.RegionName == regionName[region-1]]['TotalPositiveCases'] , x = regions['Date'], label = regionName[region-1]).set_title('Most growth cases regions')