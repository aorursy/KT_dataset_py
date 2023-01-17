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
#Load the data from /pm25-mean-annual-exposure/PM25_MAE.csv
#Create a DataFrame with the csv file. /
df = pd.read_csv('../input/pm25-mean-annual-exposure/PM25_MAE.csv', index_col=0)
#drop the column of "Country Code", "Indicator Name" and "Indicator Code"
df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
#There is no data for any country for some of the years(column), thus we drop those column by setting a threshold of 10.
df = df.dropna(thresh = 10, axis=1)
before = df.shape[0]
#There are some countries don't have data reported, we drop those countries(row)
na_free = df.dropna(thresh = 10, axis=0)
#store the only na columns to present the country name of which don't report PM2.5 data.
only_na = df[~df.index.isin(na_free.index)]
after = na_free.shape[0]
print(str(before - after) + " countries don't have any PM2.5 data reported:")

namelist = only_na.index
#print out all countries that don't have PM2.5 report.
for name in namelist:
    print(name)
df = na_free.transpose()
import matplotlib.pyplot as plt
import datetime as dt
#tester, fetch world's average data 
df.plot(y = 'World')
#Extract data from World and China to compare
df[["World", "China"]].plot(kind="bar")
#fetch the latest data for three columns and draw a pie chart out of two
chinauslatest = df.tail(1)[["China", "United States"]].transpose()
#compare 1990 and 2017 data.
chinauslatest["2017"].plot.pie()