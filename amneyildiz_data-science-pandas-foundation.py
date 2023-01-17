# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
df.head()
lesson = ["mathematics","fizik"]
note = ["80","80"]
list_label = ["lesson","note"]
list_col = [lesson,note]
zipped = list(zip(list_label,list_col))
df_dict = dict(zipped)
df = pd.DataFrame(df_dict)
df
# Add new columns
df["passing state"] = ["passed","passed"]
df
df["income"] = "BB" #Broadcasting entire column
df
# Plotting all data 
df1 = df.loc[:,["Attack","Defense","Speed"]]
df1.plot()
# it is confusing
# subplots
df1.plot(subplots = True)
plt.show()
# scatter plot  
df1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()
# hist plot  
df1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,350),normed = True)
df.describe()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
df2 = df.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
df2["date"] = datetime_object
# lets make date as index
df2= df2.set_index("date")
df2 
# Now we can select according to our date index
print(df2.loc["1993-03-16"])
print(df2.loc["1992-03-10":"1993-03-16"])
df2.resample("A").mean()
df2.resample("M").mean()
df2.resample("M").first().interpolate("linear")
df2.resample("M").mean().interpolate("linear")
