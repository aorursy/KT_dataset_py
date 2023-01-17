# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = '../input/weatherdataset1/weatherdataset1.csv.csv'
df = pd.read_csv(file_path)

df.shape #96453 records and 12 columns
df.dtypes
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

df['Formatted Date']

df.dtypes
df = df.set_index('Formatted Date')

df.head()

data_columns = ['Apparent Temperature (C)', 'Humidity']

df_monthly_mean = df[data_columns].resample('MS').mean()

df_monthly_mean.head()

#Plotting the variation in Apparent Temperature and Humidity with time
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(20,12))

plt.title("Variation in Apparent Temperature and Humidity with time")

sns.lineplot(data=df_monthly_mean)
df1 = df_monthly_mean[df_monthly_mean.index.month==4]

print(df1)

df1.dtypes