# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/airports.csv')
data.head()
data.columns
#Building Data Frames From Scratch
car_brand=["Audi","Mercedes","Bmw"]
price=["50","100","120"]
date=["2016","2017","2018"]
list_label=["car_brand","price","date"]
list_col=[car_brand,price,date]
zipped=list(zip(list_label,list_col))
data_dict=dict(zipped)
df=pd.DataFrame(data_dict)
df
df["km"]=["3570","6305","7005"]
df
df["color"]="black"
df
#Visual Exploratory Data Analysis
data1=data.loc[:,["LATITUDE","LONGITUDE"]]
data1.plot() 
data1.plot(subplots=True)
plt.show()
data1.plot(kind="scatter",x="LATITUDE",y="LONGITUDE")
plt.show()
data1.plot(kind="hist",y="LATITUDE",bins=50,range=(0,50),normed=True)
fig,axes=plt.subplots(nrows=2,ncols=1)
data1.plot(kind="hist",y="LATITUDE",bins=50,range=(0,50),normed=True,ax=axes[0])
data1.plot(kind="hist",y="LATITUDE",bins=50,range=(0,50),normed=True,ax=axes[1],cumulative=True)
plt.savefig('graph.png')
plt
#Statistical Exploratory Data Analysis
data.describe()
#Indexing Pandas Time Series
time_list=["1992-10-12","1990-09-06"]
print(type(time_list))
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")
data2=data.head()
date_list=["1992-10-05","1992-08-02","1992-06-25","1993-10-01","1993-08-30"]
datetime_object=pd.to_datetime(date_list)
data2["date"]=datetime_object
data2=data2.set_index("date")
data2
print(data2.loc["1993-08-30"])
print(data2.loc["1992-10-05":"1993-08-30"])
#Resampling Pandas Time Series
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")