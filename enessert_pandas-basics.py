# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
# data frames from dictionary
country = ["Turkey","USA"]
population =["10000","15000"]
list_label= ["country","population"]
list_col =[country,population]
zipped =list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
             
             
# Add column 
df["capital"] =["ankara","miami"]
df
#Broadcasting
df["income"]=0 #Broadcasting entire column
df
# Plotting all data
new_data =data.loc[:,["runtime","popularity"]]
new_data.plot()
# it is confusing
#subplots
new_data.plot(subplots=True)
plt.show()

# scatter plot
new_data.plot(kind="scatter",x="runtime",y="popularity")
plt.show()
#histogram plot
new_data.plot(kind="hist",y="popularity",bins=40,range =(0,250),normed=True)

# histogram subplot with non-cumulative and cumulative
fig,axes =plt.subplots(nrows=2,ncols=1)
new_data.plot(kind="hist",y="popularity",bins =50,range =(0,250),normed=True,ax=axes[0])
new_data.plot(kind="hist",y="popularity",bins =50,range =(0,250),normed=True,ax=axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
time_list = ["2001-05-07","2001-04-11"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
# In order to practice lets take head of tmdb_500 data and add it a time list
data1 =data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object =pd.to_datetime(date_list)
data1["date"] =datetime_object
# lets make date as index
data1 =data1.set_index("date")
data1
# Now we can select according to our date index
print(data1.loc["1993-03-16"])
print(data1.loc["1992-02-10":"1993-03-16"])
# We will use data1 that we create at previous part
data1.resample("A").mean()
# Lets resample with month
data1.resample("M").mean()
# As you can see there are a lot of nan because data1 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data1.resample("m").first().interpolate("linear")
# Or we can interpolate with mean()
data1.resample("m").mean().interpolate("linear")
