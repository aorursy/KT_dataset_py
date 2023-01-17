# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas import datetime
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/temperature-readings-iot-devices/IOT-temp.csv")
data.head()
#Checking Null values
data.isna().sum()
# Changing "Noted_Date" to datetime field format

data['noted_date'] = pd.to_datetime(data['noted_date'])
data.dtypes
data['Month'] = data.noted_date.dt.month
data['Year'] = data.noted_date.dt.year
data['Day'] = data.noted_date.dt.day
data_out = data[data['out/in'] == 'Out']
data_in = data[data['out/in'] == 'In']
# Segregate date field into years, months, days for more easy plotting, and add color to plots
import matplotlib.pyplot as plt

plt.scatter(data_in.Month,data_in.temp,color="b")
plt.scatter(data_out.Month,data_out.temp,color="r")
plt.xlabel("Months")
plt.ylabel("Degrees in Celcius")
plt.title("Months Vs Temp (In/Out)")
plt.legend(['Inside Temperature','Outside Temperature'],fancybox=True,bbox_to_anchor=(1.04,1), loc="upper left")
