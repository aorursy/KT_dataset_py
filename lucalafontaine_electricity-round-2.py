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
df_prop = pd.read_excel("/kaggle/input/electricity-for-my-house/Propane Data (version 1).xlsx")
df_elec = pd.read_excel("/kaggle/input/electricity-for-my-house/Hydro info.xlsx")
df_elec.tail()
df_elec.dtypes
df_elec['Start date']
df_elec.iloc[[12, 14, 15, 17, 18, 21, 23, 24], 0]
pd.to_datetime(df_elec.iloc[[12, 14, 15, 17, 18, 21, 23, 24], 0], dayfirst=True)
df_time = df_elec.iloc[[12, 14, 15, 17, 18, 21, 23, 24], 0].to_frame()
df_time['Start date'] = df_time['Start date'].astype(str).str.replace('00:00:00', '')
df_time['Start date'] = pd.to_datetime(df_time['Start date'],dayfirst=True)
#df_time.map(lambda x: x.slice(_elec[col] = df_elec[col].str.replace(' KWh', '')
    
df_time.head()
#pip install --upgrade matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

df_elec.reset_index(inplace=True)
df_elec.dtypes

for i in df_elec.index:
    print(i)
df_elec["Start date"] = df_elec["Start date"].astype(str)
df_elec["Start date"].str.contains("00:00:00")
for i in df_elec["Start date"].str.contains("00:00:00"):
    if i == True:
        print (i)
    
        

#df_elec["Start date"].map(lambda x: x='hi' if x.contains("00:00:00")  )

#df_elec['Start date'] = pd.to_datetime(df_elec['Start date'])



df_elec['Start date'] = df_elec['Start date'].astype(str).str.replace('00:00:00', '')
df_elec['Start date'] = pd.to_datetime(df_time['Start date'],dayfirst=True)
#df_elec['Start date'] = df_elec['Start date'].astype('datetime64[D]')
#df_elec['End Date'] = pd.to_datetime(df_elec['End Date'])
df_elec['End Date'] = df_elec['End Date'].astype('datetime64[D]')
#df_elec.sort_values(df_elec['Start date'], axis = 0)
df_elec.tail(10)

new_cols= ['Index', 'Start', 'End', 'Days', 'Off Kwh', 
           'Mid Kwh', 'On Kwh', 'Off Cad', 'Mid Cad', 
           'On Cad', 'Delivery', 'Total Off', 'Total Mid', 
           'Total On', 'HST', 'Rebate', 'Total']

df_elec.columns = new_cols

df_elec.head(2)
for col in df_elec.select_dtypes(include='object').columns:
    df_elec[col] = df_elec[col].str.replace(' KWh', '')
    df_elec[col] = df_elec[col].str.replace(',', '.')
    df_elec[col] = df_elec[col].str.replace('$', '')
    df_elec[col] = df_elec[col].astype('float')

df_elec.head()
df_elec.dtypes
labels = ['Off-Peak', 'Mid-Peak', 'On-Peak']

fig, ax = plt.subplots()
plt.figure(figsize=(20, 15))
ax.stackplot(df_elec['Start'], df_elec['Off Kwh'],  labels=labels)
ax.legend(loc="upper left")
plt.xticks(rotation='45')
plt.show()

