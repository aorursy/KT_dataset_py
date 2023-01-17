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
# import relevant libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# reading in data from the HM land registry 
df = pd.read_csv("../input/ppd_data.csv", header=None)
df.head()
# re-naming the columns 
df.columns
df_cols = ['refnum', 'price', 'date', 'postcode', 'attribute', 'new build', 'freeholdvsleasehold', 'name', 'number', 'road', 'area', 'hassocks', 'county', 'county2', '14', 'link']
df.columns = df_cols

df.head()
df.info()
# the current data frame has 7,632 properties that have been sold, I want to subset by type of house, specifcially detached 
df_detached = df[df.attribute == 'D']
df_detached
# 2,934 properties out of the 7,632 are detached. 
# isolating the year of sale from the date 
df_dates = df_detached['date'].str.split('-', expand=True)
df_dates
# the string has been split into day, month and year
# adding these new columns back onto the data frame 
df_all = pd.concat([df_detached, df_dates], axis=1)
df_all
# re-naming the new columns 
df_cols2 = ['refnum', 'price', 'date', 'postcode', 'attribute', 'new build', 'freeholdvsleasehold', 'name', 'number', 'road', 'area', 'hassocks', 'county', 'county2', '14', 'link', 'year', 'month', 'day']
df_all.columns = df_cols2
df_all.head()
# these are currently objects so converting it to integer to it can be graphed
df_all['year'] = df_all['year'].astype(str).astype(int)
df_all.info()
# not the best way to achieve this but the only way I can think of 
# this creates a data frame for each year and the price
df1995 =  df_all[df_all.year == 1995]
df1996 =  df_all[df_all.year == 1996]
df1997 =  df_all[df_all.year == 1997]
df1998 =  df_all[df_all.year == 1998]
df1999 =  df_all[df_all.year == 1999]
df2000 =  df_all[df_all.year == 2000]
df2001 =  df_all[df_all.year == 2001]
df2002 =  df_all[df_all.year == 2002]
df2003 =  df_all[df_all.year == 2003]
df2004 =  df_all[df_all.year == 2004]
df2005 =  df_all[df_all.year == 2005]
df2006 =  df_all[df_all.year == 2006]
df2007 =  df_all[df_all.year == 2007]
df2008 =  df_all[df_all.year == 2008]
df2009 =  df_all[df_all.year == 2009]
df2010 =  df_all[df_all.year == 2010]
df2011 =  df_all[df_all.year == 2011]
df2012 =  df_all[df_all.year == 2012]
df2013 =  df_all[df_all.year == 2013]
df2014 =  df_all[df_all.year == 2014]
df2015 =  df_all[df_all.year == 2015]
df2016 =  df_all[df_all.year == 2016]
df2017 =  df_all[df_all.year == 2017]
df2018 =  df_all[df_all.year == 2018]
df_price_all = pd.concat([df1995.price, df1996.price, df1997.price, df1998.price, df1999.price, df2000.price, df2001.price, df2002.price, df2003.price, df2004.price, df2005.price, df2006.price, df2007.price, df2008.price, df2009.price, df2010.price, df2011.price, df2012.price, df2013.price, df2014.price, df2015.price, df2016.price, df2017.price, df2018.price], axis=1)
df1995
df1995 =  df_all[df_all.year == 1995]
df1995
df_all
df_price_all
df_cols3 = '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'
df_price_all.columns = df_cols3
df_price_all.head()
df_mean = df_price_all.mean(axis=0)
df_mean
