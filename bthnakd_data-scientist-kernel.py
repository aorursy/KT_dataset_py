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
data = pd.read_csv("../input/googleplaystore.csv",encoding='ISO-8859-1')
data
data.info()
data['PointOnATenPointScale'] = data['Rating'] + data['Rating']
data
data.head(10)
data.tail(10)
data.columns
data.shape
#EXPLORATORY DATA ANALYSIS
print(data["Category"].value_counts(dropna = False))
data.describe()
data.boxplot(column = "Rating", by = "Content Rating",figsize = (15,15))
#TIDY DATA
dataNew = data.head()
dataNew
meltedData = pd.melt(frame = dataNew,id_vars = "App", value_vars = ["Size","Installs"])
meltedData
#PIVOTING DATA
meltedData.pivot(index = "App", columns = "variable", values = "value")
#CONCATENATING DATA
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis = 0, ignore_index = True)
conc_data_row
data1 = data["Current Ver"].head()
data2 = data["Android Ver"].head()
conc_data_columns = pd.concat([data1,data2],axis = 1, ignore_index = True)
conc_data_columns
#DATA TYPES
data.dtypes
data["App"] = data["App"].astype("category")
data["Category"] = data["Category"].astype("category")
data["PointOnATenPointScale"] = data["PointOnATenPointScale"].astype("float")
data.dtypes
#MISSING DATA AND TESTING WITH ASSERT
data.info()
data["Rating"].value_counts(dropna = False)
data1 = data
data1["Rating"].dropna(inplace = True)
assert data["Rating"].notnull().all()
data["Rating"].fillna("empty",inplace = True)
assert data["Rating"].notnull().all()
data.info()
data["Rating"].value_counts(dropna = False)
Country = ["Turkey","Germany"]
Population = ["32","34"]
listLabel =["Country","Population"]
listCol = [Country,Population]
zippedData = list(zip(listLabel,listCol))
data_dict = dict(zippedData)
df = pd.DataFrame(data_dict)
df
df["Capital"] = ["Ankara","Berlin"]
df
df["Islands"] = 0
df
data1 = data.loc[:,["Rating","PointOnATenPointScale"]]
data1.plot()
data1.plot(subplots = True)
plt.show()
data1.plot(kind = "scatter",x = "PointOnATenPointScale",y = "Rating")
plt.show()
data1.plot(kind = "hist",y = "PointOnATenPointScale",bins = 50,range = (0,10),normed = True)
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "PointOnATenPointScale",bins = 50,range= (0,10),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "PointOnATenPointScale",bins = 50,range= (0,10),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
time_list = ["2018-03-08","2018-04-12"]
print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")

data2 = data.head()
date_list = ["2018-01-10","2018-02-10","2018-03-10","2018-03-15","2018-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["Date"] = datetime_object

data2 = data2.set_index("Date")
data2 
print(data2.loc["2018-03-16"])
print(data2.loc["2018-03-10":"2018-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv("../input/googleplaystore.csv",encoding='ISO-8859-1')
#data = data.set_index("#")
#data.index = data["#"]
data.head()
data["App"][1]
data.App[1]
data.loc[1,["App"]]
data[["App","Reviews"]]
print(type(data["App"]))     
print(type(data[["App"]]))
data.loc[1:10,"App":"Price"]
data.loc[10:1:-1,"App":"Price"]
data.loc[1:10,"Rating":] 
boolean = data.Rating > 3.9
data[boolean]
first_filter = data.Rating > 3.9
second_filter = data.Type != "Free"
data[first_filter & second_filter]
data.App[data.Genres == "Business"]
data['PointOnATenPointScale'] = data['Rating'] + data['Rating']
def div(n):
    return n/5
data.PointOnATenPointScale.apply(div)
data.Rating.apply(lambda n : n*3)
data["RatingAverage"] = (data.Rating + data.PointOnATenPointScale)/2
data.head()
print(data.index.name)
data.index.name = "IndexName"
data.head()
data.head()
data3 = data.copy()
data3.index = range(100,10941 ,1)
data3.head()
data1 = pd.read_csv("../input/googleplaystore.csv",encoding='ISO-8859-1')
data1.head(100)
data1 = data.set_index(["Category","App"]) 
data1.head(100)
data1.loc["BEAUTY","Hush - Beauty for Everyone"]
dic = {"car":["A","A","B","B"],"price":[40.000,45.000,50.000,60.000],"receiver":["F","M","M","F"],"model":["2010","2011","2012","2013"]}
df = pd.DataFrame(dic)
df
df.pivot(index="car",columns = "receiver",values="model")
df1 = df.set_index(["car","receiver"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
df
pd.melt(df,id_vars="car",value_vars=["receiver","price"])
df
df.groupby("car").mean()
df.groupby("car").price.max() 
df.groupby("car")[["price","model"]].min() 
df.info()
# If we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
df["car"] = df["car"].astype("category")
df["model"] = df["model"].astype("category")
df.info()