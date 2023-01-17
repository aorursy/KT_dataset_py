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
import pandas as pd
import numpy as np

df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
df.shape
df.isnull().sum()
for x in ["SO2","NO2","O3","CO","PM10","PM2.5"]:
    print(x+" : ")
    print(df[x].describe())
for x in ["SO2","NO2","O3","CO","PM10","PM2.5"]:
    df=df[df[x]>0]
df.describe()
df[df["Station code"]==101]
del df["Latitude"]
del df["Longitude"]
del df["Address"]
def normalize_data(x):
    index=[]
    
    for date in x:
        normalized= date.split(' ')
        normalized= normalized[0].split('-')+normalized[1].split(':')
        index.append(normalized)
        
    
    return pd.DataFrame(index,columns=["Year","Month","Day","Hour","Min"])
df_data_normalized=normalize_data(df["Measurement date"])
print(f'Normalized date shape {df_data_normalized.shape}.')
print(f'Original Df shape {df.shape}.')

df_normalized=pd.concat([df,df_data_normalized],axis=1)
print(f'Final shape {df_normalized.shape}')
df_normalized
df_final=df_normalized.dropna()
del df_final['Measurement date']
df_final
df_final['Min'].value_counts()

del df_final['Min']
df_final
