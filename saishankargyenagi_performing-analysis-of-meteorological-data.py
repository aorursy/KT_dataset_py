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
path = '../input/weather-dataset/weatherHistory.csv'

df = pd.read_csv(path)
df.shape #96453 records and 12 columns
df.describe
df.dtypes
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Formatted Date']
df.dtypes
df = df.set_index('Formatted Date')
df.head()
data_columns = ['Apparent Temperature (C)', 'Humidity']
df_monthly_mean = df[data_columns].resample('MS').mean()
df_monthly_mean.head()
#Quantitative variables:
quantitative = df.select_dtypes(include = ["int64","float64"]).keys()
print(quantitative)
df[quantitative].describe()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(10,6))
plt.title("Variation in Apparent Temperature and Humidity with time")
sns.lineplot(data=df_monthly_mean)
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 9, 9
df[quantitative].hist()
rcParams['figure.figsize'] = 8, 5
sns.countplot(y=df['Summary'])
df1 = df_monthly_mean[df_monthly_mean.index.month==4]
print(df1)
df1.dtypes

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)')
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Humidity'], marker='o', linestyle='-',label='Humidity')

ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')
df = df.reset_index()
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],utc = True)
df['month'] = df['Formatted Date'].dt.month
df['year'] = df['Formatted Date'].dt.year
data3 = df.groupby(['month','year']).mean()
data3
plt.figure(figsize=(15,10))
a = np.arange(2006,2017)
plt.plot(a,data3['Apparent Temperature (C)'][:11].values, label = 'Jan. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][11:22].values, label = 'feb. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][22:33].values, label = 'march. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][33:44].values, label = 'april change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][44:55].values, label = 'may change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][55:66].values, label = 'June change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][66:77].values, label = 'July change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][77:88].values, label = 'aug. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][88:99].values, label = 'sep. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][99:110].values, label = 'oct. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][110:121].values, label = 'nov. change in Apparent temp.')
plt.plot(a,data3['Apparent Temperature (C)'][121:132].values, label = 'dec. change in Apparent temp.')

plt.legend()
plt.title('Variation in Apparent temp. in different months',fontsize = 15)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.figure(figsize=(15,10))
a = np.arange(2006,2017)
plt.plot(a,data3['Humidity'][:11].values, label = 'Jan. change in Humidity')
plt.plot(a,data3['Humidity'][11:22].values, label = 'feb. change in Humidity')
plt.plot(a,data3['Humidity'][22:33].values, label = 'march. change in Humidity')
plt.plot(a,data3['Humidity'][33:44].values, label = 'april Humidity')
plt.plot(a,data3['Humidity'][44:55].values, label = 'may Humidity')
plt.plot(a,data3['Humidity'][55:66].values, label = 'June Humidity')
plt.plot(a,data3['Humidity'][66:77].values, label = 'July Humidity')
plt.plot(a,data3['Humidity'][77:88].values, label = 'aug. change in Humidity')
plt.plot(a,data3['Humidity'][88:99].values, label = 'sep. change in Humidity')
plt.plot(a,data3['Humidity'][99:110].values, label = 'oct. change in Humidity')
plt.plot(a,data3['Humidity'][110:121].values, label = 'nov. change in Humidity')
plt.plot(a,data3['Humidity'][121:132].values, label = 'dec. change in Humidity')
plt.legend()
plt.title('Variation of Humidity in different months',fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
