# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/weather-dataset/weatherHistory.csv')
df.head()
df.describe()
df = df.drop(['Daily Summary','Wind Bearing (degrees)','Summary','Precip Type','Temperature (C)',
              'Loud Cover','Wind Speed (km/h)','Visibility (km)','Pressure (millibars)'], axis = 1)
df.head()
df.isnull().sum()
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],utc=True)
df = df.set_index('Formatted Date')
data = df[['Apparent Temperature (C)','Humidity']].resample('MS').mean()
data
plt.figure(figsize=(15,3));
plt.plot(data['Humidity'], label = 'Humidity', color = 'orange',linestyle='dashed');
plt.plot(data['Apparent Temperature (C)'], label = 'Apparent temp.',color = 'green');
plt.title('Variation of Apparent temparature v/s Humidity', fontsize= 25);
plt.legend(loc = 0, fontsize = 12);
plt.xticks(fontsize = 15);
plt.yticks(fontsize = 13);
april = data[data.index.month==4]
plt.figure(figsize=(15,3))
plt.plot(april.loc['2006-04-01':'2016-04-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(april.loc['2006-04-01':'2016-04-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of April', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
jan = data[data.index.month==1]
plt.figure(figsize=(15,5))
plt.plot(jan.loc['2006-01-01':'2016-01-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(jan.loc['2006-01-01':'2016-01-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of January', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
feb = data[data.index.month==2]
plt.figure(figsize=(15,5))
plt.plot(feb.loc['2006-02-01':'2016-02-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(feb.loc['2006-02-01':'2016-02-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of feb', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
march = data[data.index.month==3]
plt.figure(figsize=(15,5))
plt.plot(march.loc['2006-03-01':'2016-03-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(march.loc['2006-03-01':'2016-03-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of March', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
may = data[data.index.month==5]
plt.figure(figsize=(15,5))
plt.plot(may.loc['2006-05-01':'2016-05-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(may.loc['2006-05-01':'2016-05-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of May', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
june = data[data.index.month==6]
plt.figure(figsize=(15,5))
plt.plot(june.loc['2006-06-01':'2016-06-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(june.loc['2006-06-01':'2016-06-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of June', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
july = data[data.index.month==7]
plt.figure(figsize=(15,5))
plt.plot(july.loc['2006-07-01':'2016-07-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(july.loc['2006-07-01':'2016-07-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of July', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
aug = data[data.index.month==8]
plt.figure(figsize=(15,5))
plt.plot(aug.loc['2006-08-01':'2016-08-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(aug.loc['2006-08-01':'2016-08-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of August', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
sept = data[data.index.month==9]
plt.figure(figsize=(15,5))
plt.plot(sept.loc['2006-09-01':'2016-09-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(sept.loc['2006-09-01':'2016-09-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of September', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
octo = data[data.index.month==10]
plt.figure(figsize=(15,5))
plt.plot(octo.loc['2006-10-01':'2016-10-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(octo.loc['2006-10-01':'2016-10-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of October', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
nov= data[data.index.month==11]
plt.figure(figsize=(15,5))
plt.plot(nov.loc['2006-11-01':'2016-11-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(nov.loc['2006-11-01':'2016-11-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of November', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
dec = data[data.index.month==12]
plt.figure(figsize=(15,5))
plt.plot(dec.loc['2006-12-01':'2016-12-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)',color = 'green');
plt.plot(dec.loc['2006-12-01':'2016-12-01', 'Humidity'], marker='o', linestyle='-',label='Humidity',color = 'orange');
plt.legend(loc = 'center right',fontsize = 15);
plt.xlabel('Month of December', fontsize = 15);
plt.title('Humidity v/s Apparent Temperature',fontsize = 15)
