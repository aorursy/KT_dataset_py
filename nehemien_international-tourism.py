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



import matplotlib.pyplot as plt

import datetime as dt

import pandas as pd
# import datasets

arrival_df = pd.read_csv("/kaggle/input/international-tourism-number-of-arrivals/API_ST.INT.ARVL_DS2_en_csv_v2_888001.csv")

depart_df = pd.read_csv("/kaggle/input/tourism-data-v1/ITour_nbr of departures.csv")

co2_df = pd.read_csv ("/kaggle/input/tourism-data-v1/Co2 Emissions.csv")


#arrival data cleaning

#create a deta frame that has Japan as a column and set 'Year' as index

arrival_df.set_index("Country Name",inplace=True)

arrival=arrival_df.loc [['Japan'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

arrival.columns.name = None

arrival.index.names = ['Year']



#departure data cleaning

depart_df.set_index("Country Name",inplace=True)

depart=depart_df.loc [['Japan'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

depart.columns.name = None

depart.index.names = ['Year']



#co2 emission data cleaning and manipulation

#create a deta frame that has France, Japan and Australia as columns and set 'Year' as index

co2_df.set_index("Country Name",inplace=True)

co2=co2_df.loc [['Japan'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

co2.columns.name = None

co2.index.names = ['Year']



#create a dataframe that includes both departure and arrival data

#set 'Arrival' and 'Departure' as columns

#set multi-index sorted by country and then by year

traffic1 = pd.concat([arrival,depart], axis=1,sort=False, keys=['Arrival','Departure']).stack()

traffic = traffic1.swaplevel(0, 1, axis=0).sort_index()

traffic.index.set_names(['Country', 'Year'], inplace=True)

traffic.head()
#visualize the data frame that only contains arrival counts

#use subplot to show the three diagrams separately

arrival.plot.barh(colors=['red','orange','deeppink'], figsize=(8, 6), legend=True,subplots=True)
# compare the number of arrival and departure by plotting each country's data into stacked bar diagrams

colors = ['red','orange']



bar_figsize = (10, 4)



traffic.loc['Japan'].plot.bar(rot=0, figsize=bar_figsize, stacked=True,title='Japan: Arrival vs. Departure',colors=colors)

#create a dataframe that includes both arrival and emission data

#set 'Arrival' and 'Emission' as columns

#set multi-index sorted by country and then by year

traffic1 = pd.concat([arrival,co2], axis=1,sort=False, keys=['Arrival','Emission']).stack()

trafficem = traffic1.swaplevel(0, 1, axis=0).sort_index()

trafficem.index.set_names(['Country', 'Year'], inplace=True)

trafficem.head()
# compare the trend in arrival number and co2 emission by plotting each index value into two line diagrams: one for arrival and one for emission

colors = ['red','orange']

default_figsize = (8,8)



trafficem.loc['Japan'].plot(figsize= default_figsize,stacked=True,title='Japan',linewidth = 2, colors=colors,subplots=True)

#create a data frame that only has arrival data of 'world' 

arrivalglb=arrival_df.loc [['World'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

arrivalglb.columns.name = None

arrivalglb.index.names = ['Year']



#create a data frame that only has emission data of 'world'

co2_global=co2_df.loc [['World'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

co2_global.columns.name = None

co2_global.index.names = ['Year']



#combine the two data frames

traffic_sec = pd.concat([arrivalglb,co2_global], axis=1,sort=False, keys=['Arrival','Emission']).stack()

traffic_global = traffic_sec.swaplevel(0, 1, axis=0).sort_index()

traffic_global.index.set_names(['Country', 'Year'], inplace=True)

traffic_global.loc['World'].plot(figsize=(8, 8),stacked=True,linewidth = 2,title='World',colors=colors,subplots=True)