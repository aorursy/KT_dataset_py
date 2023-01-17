# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
# reading in data from the HM land registry 
df = pd.read_csv("../input/ppd_data.csv", header=None)
df.head()
# naming the columns 
df.columns
df_cols = ['refnum', 'price', 'date', 'postcode', 'attribute', 'new build', 'freeholdvsleasehold', 'name', 'number', 'road', 'area', 'hassocks', 'county', 'county2', '14', 'link']
df.columns = df_cols
df.head()
#subset by detached properties only
df_detached = df[df.attribute == 'D']
df_detached.info()
#split the date column into 3 seperate columns - to get years on its own
df_dates = df_detached['date'].str.split('/', expand=True)
df_dates
#merge the dataframes back together 
df_all = pd.concat([df_detached, df_dates], axis=1)
df_all.head()
df_all.columns
# rename final columns to get rid of number names 
df_cols2 = ['refnum', 'price', 'date', 'postcode', 'attribute', 'new build', 'freeholdvsleasehold', 'name', 'number', 'road', 'area', 'hassocks', 'county', 'county2', '14', 'link', 'day', 'month', 'year']
df_all.columns = df_cols2
df_all.head()
# find out what possible years houses were sold? 
df_all.year.unique()
# convert this from an object to a integer so it can be graphed 
# df_all['year'].astype(str).astype(int)
# this keeps the column in the dataframe 
df_all['year'] = df_all['year'].astype(str).astype(int)
df_all.info()
# year is now an integer
df_all.head()

df_all['year']
hist_plot = df_all['year'].hist(bins=23)
# this appears to show a large increase in detached properties sold in 2018 for this particular town - will need to investigate further 
#detached properties on land registry from 1995 to 2018 

