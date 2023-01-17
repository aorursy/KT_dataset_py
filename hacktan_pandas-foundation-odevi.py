# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data=pd.read_csv("../input/Iris.csv")

# Any results you write to the current directory are saved as output.
# data frames from dictionary
country=["Franche","England"]
population=["25","28"]
list_label=["Country","Population"]
list_col=[country,population]
zipped=list(zip(list_label,list_col))
data_dict=dict(zipped)
db=pd.DataFrame(data_dict)
print(db)
#adding new column
db["newcol"]=[3,5]
db
print(db)
db["newcol"]=0
print(db)
data
#plotting all data (confused)
data1 = data.loc[:,["SepalLengthCm","PetalLengthCm"]]
data1.plot()
plt.show()
#subplots
data1.plot(subplots=True)
plt.show()
data1.plot(kind="scatter",x = "SepalLengthCm",y = "PetalLengthCm")
plt.show()
#hist plot
data1.plot(kind="hist",bins=50,y="SepalLengthCm",range=(0,8),normed=True)
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "SepalLengthCm",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "PetalLengthCm",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt

data.describe()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1]))
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
# In order to practice lets take head of Irıs data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object=pd.to_datetime(date_list)
data2["date"]=datetime_object
data2=data2.set_index("date")
data2
# Now we can select according to our date index
#print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
#print(data2)
# We will use data2 that we create at previous part
data2.resample("A").mean()

# Lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")