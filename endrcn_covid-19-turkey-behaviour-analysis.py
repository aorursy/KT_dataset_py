# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.figure as fig



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
df.columns
# Calculate the Daily Increasing

df.sort_values(['Country/Region', 'ObservationDate'], ascending=[True, True], inplace=True)

turkeyFilter = df['Country/Region'] == 'Turkey'

TRData = df[turkeyFilter].copy()

TRData["DailyIncrease"] = TRData['Confirmed'].diff().fillna(0)

TRData["DailyDeaths"] = TRData['Deaths'].diff().fillna(0)

TRData["DailyRecovered"] = TRData['Recovered'].diff().fillna(0)

TRData
plt.figure(figsize=(40,20))

plt.legend()

plt.grid(True)

plt.plot(TRData.ObservationDate, TRData.DailyIncrease, color='blue', linewidth=1.0)

plt.plot(TRData.ObservationDate, TRData.DailyDeaths, color='red', linewidth=1.0)

plt.plot(TRData.ObservationDate, TRData.DailyRecovered, color='green', linewidth=1.0)

plt.ylabel("Numbers")

plt.xlabel("Dates")

plt.xticks(rotation=90)

plt.title("Daily Numbers for Turkey")

plt.show()
TRData["ActiveCount"] = TRData['Confirmed'] - TRData['Recovered'] - TRData['Deaths']

plt.figure(figsize=(40,20))

plt.grid(True)

plt.plot(TRData.ObservationDate, TRData.ActiveCount, color='green', linewidth=1.0)

plt.ylabel("Active Count")

plt.xlabel("Dates")

plt.xticks(rotation=90)

plt.title("Daily Active Counts of Turkey")

plt.show()
print("MAX",TRData.ActiveCount.max())

print("AVG",TRData.ActiveCount.mean())

print("MIN",TRData.ActiveCount.min())