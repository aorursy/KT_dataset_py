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
df = pd.read_csv('/kaggle/input/30_day_parking_2020_05_09.csv')

df
#Create a list of all the rows where the category is not Paid Parking

indexNotPaid = df[df['ParkingCategory'] != "Paid Parking"].index

indexNotPaid
#Now drop all those rows.

# Delete these row indexes from dataFrame

df.drop(indexNotPaid , inplace=True)

df
if ("SideOfStreet" in df.columns): #really simple check so I can leave this drop un-commented and have this line run when I run the notebook from the begining, but not error out if I re-run this cell.

    df = df.drop(['SideOfStreet','SourceElementKey','ParkingSpaceCount','Location','BlockfaceName','PaidParkingSubArea','ParkingCategory',"PaidParkingRate","ParkingTimeLimitCategory"], axis=1)



df.head(100)
#Make a new dataframe with the date and time split out.

df[['Date','Time','AMPM']] = df.OccupancyDateTime.str.split(expand=True,)

df
#Now I am going to use the pivot table function to reshape this data so it is organized by date on the rows and the column is the parking areas.

df.pivot_table(index="Date", columns="PaidParkingArea", values="PaidOccupancy", aggfunc="sum")
#Let's save this output as a new dataframe called pivot.

pivot = pd.DataFrame(df.pivot_table(index="Date", columns="PaidParkingArea", values="PaidOccupancy", aggfunc="sum"))

pivot
#Now let's plot it.

import matplotlib.pyplot as plt

import datetime as dt



plt.style.use('ggplot')

pivot.plot()
cols = pivot.iloc[:,0:].columns #THANK YOU RAFAL - Grabs the names of all the columns.

#cols = cols.insert(0,'Date') #Add the X-axis date column

print(cols) #print the column names.



pivot[cols].plot(figsize=(12,4), subplots=True, layout=(4,5))