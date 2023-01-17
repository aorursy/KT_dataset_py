# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/home-medical-visits-healthcare/Hack2018_ES.csv')
data.columns
english_columns = ['Visit_Status', 'Time_Delay', 'City', 'City_ID', 'Patient_Age', 'Zipcode', 'LAT', 'LON', 'Pathology', 'Date', 'ID_Type',

'ID_Personal', 'N_Home_Visits', 'Is_Patient_Minor', 'Geo_Point']
data.columns = english_columns
data.head(5)
missing = []

for column in data.columns:

    missing.append({'column': column, 'missing': data[column].isnull().sum()})

missing = pd.DataFrame.from_records(missing)

missing[missing.missing > 0]
# Not a lot of rows so we can just drop them

data.dropna(inplace=True, axis=0)
import matplotlib.pyplot as plt

import seaborn as sns

print("Average wait time: {} minutes".format(data['Time_Delay'].mean()))

fig = sns.countplot(y="Time_Delay", data=data)

fig.set_title("Waiting time count")

fig.set_ylabel("Minutes")

plt.show()

fig = sns.boxplot(data['Time_Delay'])

fig.set_title("Waiting time boxplot")

fig.set_xlabel("Minutes")

plt.show()
# Converting the Date from object to Datetime

data.Date = pd.to_datetime(data.Date)

data.info()
# Indexing the Data by its Dates

data_d = data.set_index(data.Date, append=False)

data_d.head()
import datetime

subset = data_d.between_time("20:13:00", "20:18:00")
subset
def split_date(df):

    df = df.copy()

    df['Year'] = pd.DatetimeIndex(df['Date']).year

    df['Month'] = pd.DatetimeIndex(df['Date']).month

    df['Day'] = pd.DatetimeIndex(df['Date']).day

    df['Time'] = pd.DatetimeIndex(df['Date']).time

    return df

data_s = split_date(data)

data_s.head(5)
### You can filter more freely:

data_s[data_s['Year'] == 2017][data_s['Month'] == 6][data_s['Day'] == 21][data_s['Time'] <= datetime.time(17,25,30)]