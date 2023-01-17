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
import datetime

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
italy_weather = pd.read_csv('/kaggle/input/covid-19-weather-dataset/italy_weather.csv')
italy_weather.head()
fig = plt.figure(figsize=(20,10))

sns.distplot(italy_weather.Temperature)

plt.title("Distribution of climate in Ireland")
italy_weather.plot(x='Date',y='Temperature',style='o')  

plt.title('Temperature variation over a period of time')  

plt.xlabel('Date')  

plt.ylabel('Temperature')  

plt.show()
plt.figure(figsize=(30,10))

plt.title('Tempearature variation in Ireland')

plt.xlabel('Date')

plt.ylabel('Temperature')

plt.plot(italy_weather.Date,italy_weather.Temperature)

plt.legend(['Temperature'])