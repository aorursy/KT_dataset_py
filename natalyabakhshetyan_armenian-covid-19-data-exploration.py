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
df = pd.read_csv("/kaggle/input/covid19-in-armenia/arm_covid_with_calculated_columns.csv")
df.head()
df.shape
df.info()
#convert date type from string to datetime
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df.head()
#create a column with death rate calculation
df['death_rate'] = (df['total_deaths']*100/(df['total_cases'] + df['total_deaths'])).round(2)
df
df.describe()
plt.figure(figsize = (10,7))
plt.plot(df.date, df.percent_positive)
plt.xlabel('date')
plt.ylabel('% of positive cases among all tests')
plt.title('% of positive cases among all tests for each day')
plt.figure(figsize = (10,7))
plt.plot(df.date, df.deaths)
plt.xlabel('date')
plt.ylabel('deaths')
plt.title('Number of deaths')
plt.figure(figsize = (10,7))
plt.plot( 'date', 'confirmed_cases', data=df)
plt.plot( 'date', 'recovered', data=df)
plt.plot( 'date', 'deaths', data=df)
plt.xlabel('date')
plt.legend()

df.columns
plt.figure(figsize = (10,7))
plt.plot( 'date', 'total_cases', data=df)
plt.plot( 'date', 'total_recovered', data=df)
plt.plot( 'date', 'total_tests', data=df)
plt.plot( 'date', 'total_deaths', data=df)
plt.xlabel('date')
plt.legend()
plt.figure(figsize = (10,7))
plt.plot(df.date, df.death_rate)
plt.xlabel('Date')
plt.ylabel('Death Rate(%)')
plt.title('Death Rate')