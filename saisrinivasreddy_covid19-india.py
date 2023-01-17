# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
from urllib.request import urlopen

import json

import pandas as pd

from pandas.io.json import json_normalize



import matplotlib.pyplot as plt

import seaborn as sns 



import warnings



warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

# sns.set_palette("Paired")
response = urlopen('https://api.covid19india.org/raw_data.json')

elevations = response.read()

data = json.loads(elevations)

js = json_normalize(data['raw_data'])

js.to_csv('covid19ind.csv',index=False)
df = pd.read_csv('covid19ind.csv',parse_dates=['dateannounced'],dayfirst=True)

df.dropna(subset=['dateannounced'], inplace=True)
df[df['detectedstate']=='Maharashtra']
#df.set_index('dateannounced')

print(df.shape)

df.head()
plt.figure(figsize=(15,10))

ax = df['detectedstate'].value_counts().plot('barh',colormap='prism')

ax.invert_yaxis()

plt.title('COVID19-India (State wise)')
sdf = df[['detectedstate','patientnumber','dateannounced']]

sdf['count'] = 1

sdf.info()
# ap = sdf[sdf.detectedstate=='Andhra Pradesh']

gdf = sdf.groupby(['dateannounced','detectedstate'])['count'].sum()
plt.figure(figsize=(15,10))

sdf.groupby(['dateannounced'])['count'].sum().plot(marker='o')
fdf = gdf.unstack(level = 'dateannounced',fill_value=0)
cumdf = fdf.cumsum(axis=1)

cumdf.columns = cumdf.columns.strftime('%d-%m')
cumdf.head()
len(cumdf.columns)
cols = []

for i in range(1,len(cumdf.columns)+1):

    cols.append("Day: "+str(i))    
cumdf.columns = cols
cumdf.head()
cumdf.to_csv('covid19-india.csv')