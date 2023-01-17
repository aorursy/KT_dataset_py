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
#Reading file
file = open('/kaggle/input/turkey-corona-data/turkey_corona_data.csv','r')
df = pd.read_csv(file)
df
df.info()
df.isnull().sum()
df.describe()
# Total Recovered
df['Recovered'].tail(1)
#Total Deaths
df['Deaths'].tail(1)
#Total Confirmed
df['Confirmed'].tail(1)
import cufflinks as cf
cf.go_offline()
df.iplot(title="Cases to Death Graph",xTitle='Death',yTitle='Confirmed',kind='scatter',x='Deaths',y='Confirmed')
df.iplot(title="Cases to Death Graph",xTitle='Recovered',yTitle='Confirmed',kind='scatter',x='Recovered',y='Confirmed')
