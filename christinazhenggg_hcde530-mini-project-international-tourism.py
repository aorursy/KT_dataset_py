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

arrivalraw = pd.read_csv("../input/internatioanltraveling/International tourism_number of arivals.csv")

departraw = pd.read_csv("../input/internatioanltraveling/International tourism_number of departures.csv")

co2raw = pd.read_csv ("../input/internatioanltraveling/CO2_Emission.csv")



#arrival data cleaning

#create a deta frame that has France, Japan and Australia as columns and set 'Year' as index

arrivalraw.set_index("Country Name",inplace=True)

arrival=arrivalraw.loc [['France','Japan','Australia'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

arrival.columns.name = None

arrival.index.names = ['Year']



#departure data cleaning

#create a deta frame that has France, Japan and Australia as columns and set 'Year' as index

departraw.set_index("Country Name",inplace=True)

depart=departraw.loc [['France','Japan','Australia'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

depart.columns.name = None

depart.index.names = ['Year']



#co2 emission data cleaning and manipulation

#create a deta frame that has France, Japan and Australia as columns and set 'Year' as index

co2raw.set_index("Country Name",inplace=True)

co2=co2raw.loc [['France','Japan','Australia'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

co2.columns.name = None

co2.index.names = ['Year']



#create a dataframe that includes both departure and arrival data

#set 'Arrival' and 'Departure' as columns

#set multi-index sorted by country and then by year

traffic1 = pd.concat([arrival,depart], axis=1,sort=False, keys=['Arrival','Departure']).stack()

traffic = traffic1.swaplevel(0, 1, axis=0).sort_index()

traffic.index.set_names(['Country', 'Year'], inplace=True)

traffic
#visualize the data frame that only contains arrival counts

#use subplot to show the three diagrams separately

colors1 = ['#b3b3ff','#66cc66','#ffcc00']

arrival.plot.barh(colors=colors1, figsize=(12, 18), legend=True,subplots=True)
# compare the number of arrival and departure by plotting each country's data into stacked bar diagrams

colors2 = ['#ffa64d','#339966']

traffic.loc['France'].plot.bar(rot=0, figsize=(12, 6),stacked=True,title='France: Arrival vs. Departure',colors=colors2)

traffic.loc['Japan'].plot.bar(rot=0, figsize=(12, 6),stacked=True,title='Japan: Arrival vs. Departure',colors=colors2)

traffic.loc['Australia'].plot.bar(rot=0, figsize=(12, 6),stacked=True,title='Australia: Arrival vs. Departure',colors=colors2)
#create a dataframe that includes both arrival and emission data

#set 'Arrival' and 'Emission' as columns

#set multi-index sorted by country and then by year

traffic1 = pd.concat([arrival,co2], axis=1,sort=False, keys=['Arrival','Emission']).stack()

trafficem = traffic1.swaplevel(0, 1, axis=0).sort_index()

trafficem.index.set_names(['Country', 'Year'], inplace=True)

trafficem
# compare the trend in arrival number and co2 emission by plotting each index value into two line diagrams: one for arrival and one for emission

colors3 = ['#ffa64d','#cc80ff']

trafficem.loc['France'].plot(figsize=(10, 6),stacked=True,title='France',colors=colors3,subplots=True)

trafficem.loc['Japan'].plot(figsize=(10, 6),stacked=True,title='Japan',colors=colors3,subplots=True)

trafficem.loc['Australia'].plot(figsize=(10, 6),stacked=True,title='Australia',colors=colors3,subplots=True)
#create a data frame that only has arrival data of 'world' 

arrivalglb=arrivalraw.loc [['World'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

arrivalglb.columns.name = None

arrivalglb.index.names = ['Year']



#create a data frame that only has emission data of 'world'

co2glb=co2raw.loc [['World'],['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']].T

co2glb.columns.name = None

co2glb.index.names = ['Year']



#combine the two data frames

traffic2 = pd.concat([arrivalglb,co2glb], axis=1,sort=False, keys=['Arrival','Emission']).stack()

trafficglb = traffic2.swaplevel(0, 1, axis=0).sort_index()

trafficglb.index.set_names(['Country', 'Year'], inplace=True)

trafficglb.loc['World'].plot(figsize=(14, 12),stacked=True,title='World',colors=colors3,subplots=True)