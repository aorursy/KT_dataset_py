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
from datetime import datetime,date,timedelta 
import datetime as dt
import time
import calendar
from pandas.tseries import offsets       

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath= "../input/csv-battery/Battery consumption (3min) final.csv"
battery= pd.read_csv(filepath)
battery.head()
battery.isnull().any()
df=battery['packet_time'].str.split('T',expand=True)
battery['date']= df[0]
battery['time']= df[1]
battery.drop(columns =["packet_time"], inplace = True)
battery
battery['date_time'] = battery['date'].str.cat(battery['time'], sep =" ") 
battery
pd.to_datetime(battery['date_time'])
d=[] 
for i in range(4812): 
    d2 = datetime.strptime(battery['date_time'][4811],'%Y-%m-%d %H:%M:%S.%f%z')
    d1 = datetime.strptime(battery['date_time'][i],'%Y-%m-%d %H:%M:%S.%f%z')
    dur= d2-d1                      #Then I have subtracted time in dur variable.
    dur=dur//timedelta(minutes=1)   # we need time in minutes format  
    d.append(dur)                   # all values will be added in the column d  
    i=i+1                             

timeleft=pd.DataFrame(d)

battery['time_left_in_min']=timeleft

battery
battery.to_csv(r'battery.csv')