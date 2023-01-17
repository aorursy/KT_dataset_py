# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
Brand = ["BMW","Mercedes","Audi","Toyota","Honda","Dacia"];   LuxuryLevel = [4,5,4,3,2,1];    HP = [300,270,290,175,180,110];   # making series
list_label = ["Brand","LuxuryLevel","HP"];    list_col = [Brand, LuxuryLevel, HP]; 
zipped = list(zip(list_label,list_col));   # zipping values with Labels
data_dict = dict(zipped);      # transforming our zipped list to dictionary
df = pd.DataFrame(data= data_dict);    
df
# Let's add some more columns

df['PriceRange'] = ["£££","££££","£££","££","££","£"]

df['HasRaceCar'] = [True,True,True,False,True,False]
df


# Let's Do a Broadcasting on our data
df['HasSUV'] = True;   df['HasHB'] = True;
df
df = df.append({"Brand": 'Ferrari',"LuxuryLevel": 5,"HP": 620,"PriceRange": "££££££", "HasRaceCar": True, "HasSUV": False, "HasHB": False}, ignore_index=True)
df
df.plot(kind='hist', y='HP', bins=10)
df.plot(kind='hist', y='HP', bins=10, cumulative=True)
Income = [45,50,35,45,50,80,45,50,55,72];   Tax = [5,8,4,5,9,16,5,8,9,11];  Bonus = [15,18,41,15,19,16,51,18,19,19];

labels = ["Income","Tax","Bonus"];   columns = [Income,Tax,Bonus];
zipped = list(zip(labels,columns));
data_dict = dict(zipped);

data = pd.DataFrame(data_dict);
data    # until now everything is without date.
# now we will add date on our DataFrame

date_list = ["2012-01-01","2012-01-15","2012-02-01","2012-02-15","2012-03-01","2012-03-15","2012-04-01","2012-04-15","2012-05-01","2012-05-15"]

datetime_obj = pd.to_datetime(date_list);

data["Date"] = datetime_obj;                       

data = data.set_index("Date")
data       # now we can see our dates on frame and they are our indexes.
# since we make it possible, Lets try to see our Income at February 2012..

print(data.Income["2012-02"])   # only year and month written, since we want all month.
# what about each months total ?...

data.resample("M").sum()   # M indicates the (M)onth   # .sum() shows total of whole values in the month given.
# Let's add 

data = data.append(pd.DataFrame(index=pd.DatetimeIndex(["2012-07-01"]),data={"Income":65,"Tax":9,"Bonus":11}),sort=1)
data = data.append(pd.DataFrame(index=pd.DatetimeIndex(["2012-07-15"]),data={"Income":45,"Tax":7,"Bonus":20}),sort=1)
data  # as we can see below, the 6th month do not exist on our DataFrame. we can use resampling to figure it out.
data.resample("M").first().interpolate("linear") 
# by .interpolate("linear") , we can see that our dataFrame shows us average estimated values for corresponding columns

