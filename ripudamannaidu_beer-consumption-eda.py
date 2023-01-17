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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv('../input/beer-consumption-sao-paulo/Consumo_cerveja.csv',nrows=366)
data.head(5)
data=data.dropna()
data.columns = ['date','temp_medium','temp_min','temp_max','precipitation','weekend','beer_consumption']
data.head(5)
data['temp_medium'] = data.temp_medium.str.replace(',', '.')
data['temp_min'] = data.temp_min.str.replace(',', '.')
data['temp_max'] = data.temp_max.str.replace(',', '.')
data['precipitation'] = data.precipitation.str.replace(',', '.')
data.head()
data['month']=pd.DatetimeIndex(data['date']).month
data.head()
data['temp_medium'] = data['temp_medium'].astype(float)
data['temp_min'] = data['temp_min'].astype(float)
data['temp_max'] = data['temp_max'].astype(float)
data['precipitation'] = data['precipitation'].fillna(0)
data['precipitation'] = data['precipitation'].astype(float)
data['weekend'] = data['weekend'].astype(float)

data['beer_consumption'] = data['beer_consumption'].astype(float)
data['month'] = data['month'].astype(str)
data.head()
print(data.groupby('month')['beer_consumption'].sum())

data.groupby('month').beer_consumption.sum().plot(kind='bar',legend =True)
col=np.array(data['precipitation'], np.int16)
data['precipitation']=col
def def_pre(row):
       
    if row > 50:
        return  'heavy'
        
    elif row > 10:
        return  'medium'
    elif row > 0:
        
        return  'less'
    elif row == 0:
        return  'no'
data['precipitation1'] = data['precipitation'].apply(def_pre)
data.head(5)
data.groupby('precipitation1').beer_consumption.mean().plot(kind='bar')
print(data.groupby('weekend')['beer_consumption'].sum())
sns.violinplot(data=data,x='weekend',y='beer_consumption' )
data['day']=pd.DatetimeIndex(data['date']).weekday
data.head(5)
print(data.groupby('day')['beer_consumption'].sum())
data.groupby('day').beer_consumption.sum().plot(kind='bar')
def def_temp(row):
        
    if row > 29:
        return  'hot'
        
    elif row > 22:
        return  'medium'
    elif row > 0:
        
        return  'cold'
    
data['temp_grp'] = data['temp_max'].apply(def_temp)
data.groupby('temp_grp').beer_consumption.count()
data['temp_max'].mean()
data.groupby('temp_grp').beer_consumption.mean().plot(kind='bar')

print(data.groupby('temp_grp')['beer_consumption'].sum())
