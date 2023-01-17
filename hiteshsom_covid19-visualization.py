# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "/kaggle/input/novel-corona-virus-2019-dataset"

df = pd.read_csv(f'{path}/covid_19_data.csv')
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df_grouped = df.groupby('ObservationDate', as_index=True).agg({'Confirmed': sum, 'Deaths': sum, 'Recovered': sum})
df_grouped
df_grouped['ActiveCases'] = df_grouped['Confirmed'] - df_grouped['Recovered'] - df_grouped['Deaths']
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_grouped['Confirmed'], label='Confirmed')

ax.set_xlim(xmin=min(df_grouped.index), xmax=max(df_grouped.index))

ax.ticklabel_format(axis='y', style='plain')

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels)

ax.set_title('Covid19 Cases in World')

ax.set_ylabel('Cumulative count')

ax.set_xlabel('Date')

plt.xticks(rotation=90)

plt.show()
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_grouped['Recovered'], label='Recovered', c=(66/256, 245/256, 72/256, 1))

ax.plot(df_grouped['ActiveCases'], label='Active', c=(245/256, 150/256, 66/256, 1))

ax.plot(df_grouped['Deaths'], label='Deaths', c=(245/256, 66/256, 66/256, 1))

ax.ticklabel_format(axis='y', style='plain')

ax.set_xlim(xmin=min(df_grouped.index), xmax=max(df_grouped.index))

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels)

ax.set_title('Covid19 Cases in World')

ax.set_ylabel('Cumulative count')

ax.set_xlabel('Date')

plt.xticks(rotation=90)

plt.show()