# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/pokemon.csv")
data.head()

data.columns
#creating dicts from list and zipping each other
state = ["cali","ny"]
city = ["LA","NY"]
list_label = ["state","city"]
list_col = [state,city]
zipped = list(zip(list_label,list_col)) 
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df

#adding new column 

df["population"] = [1000,5000]
df ["population"]= 0  # population after comet 
df
datax = data.loc[:,"sp_attack":"speed"]
datax.plot()
datax.plot(subplots=True)
plt.show()
data.plot(kind="scatter",x="attack",y="defense",color="orange",figsize=(9,9))
timelist = ["2222-1-12","2222-2-12"]
type(timelist[1])
dataz=data.head()
date_list = ["2222-01-10","2222-01-12","2222-02-14","2223-03-15","2223-03-24"]
datetime_obj = pd.to_datetime(date_list)
dataz["date"] = datetime_obj

#assign time series as index
dataz = dataz.set_index("date")
dataz
dataz.resample("A").mean()  # mean of the years
dataz.resample("M").mean()  #mean of th months
dataz.resample("M").first().interpolate("linear") # linearly interpolate NaN values

# Ivysaur is lost in the "name" column since both Bulbasaur & Ivysaur is in the same month.
dataz.resample("M").mean().interpolate("linear") # interpolate mean values linearly
dataz
