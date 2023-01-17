# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#building dataframe from pandas reference
d = {'Elektrik' : pd.Series([100., 102., 130.], index=['Abone 1', 'Abone 2' , 'Abone 3']),'Su' : pd.Series([150., 70., 63., 94.], index=['Abone 1', 'Abone 2' , 'Abone 3','Abone 4'])}
df = pd.DataFrame(d)
df
# buillding dataframes from DATAI's notebook
aboneler = ["Abone 1", "Abone 2" , "Abone 3","Abone 4", "Abone 5" , "Abone 6"]
d_elk=[100., 102., 130.,np.nan,104.5,75.8]
d_su = [150., 70., 94.,45,126,np.nan]
list_label = ["Aboneler","Elektrik","Su"]
list_col = [aboneler,d_elk,d_su]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
# Add new columns
df["İlçe"] = "Çankaya"
df
data=pd.read_csv('../input/FIFA18 - Ultimate Team players.csv')
data.head()
# Plotting all data 
data1 = data.loc[:,["age","height","weight"]]
data1.plot(figsize=[18,8])
plt.show()
# it is confusing
# subplots
data1.plot(subplots = True,figsize=[20,10])
plt.show()
data1 = data.loc[:,["age","height","price_pc"]]
data1.plot(kind = "scatter",x="age",y = "price_pc")
plt.show()
data.info()
data1 = data.loc[:,["age","height","overall"]]
data1.plot(figsize=[15,5],kind = "hist",y = "age",bins = 30,range= (0,40),density = True)
plt.show()
data.describe()
data.head()
time_list = ["2018-03-08","2018-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
df
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = df.head()
date_list = ["2017-11-10","2017-12-11","2018-01-09","2018-02-12","2018-05-16"]
datetime_object = pd.to_datetime(date_list)
data2["FaturaTar"] = datetime_object
# lets make date as index
data2= data2.set_index("FaturaTar")
data2 
print(data2.loc["2018-01-09"])
print(data2.loc["2018-01-09":"2018-02-12"])
data2.resample("A").median()
time_list = "2018-09-08"
datetime_object = pd.to_datetime(time_list)
df1 = pd.DataFrame([["Abone 6",410,256.8,"Çankaya"]],columns=['Aboneler','Elektrik','Su','İlçe'],index=[datetime_object])
df2 = pd.DataFrame({'Aboneler':["Abone 7"],
                    'Elektrik':[410],
                    'Su':[256.8],
                   'İlçe':["Çankaya"]},
                   index = [datetime_object])

data2=pd.concat([data2,df1,df2])
data2
data2.resample("M").mean()
data2.resample("M").mean().interpolate("linear")

data=pd.read_csv('../input/FIFA18 - Ultimate Team players.csv')
data= data.set_index("player_ID")
data.head()
# selecting cell
data["overall"][1]
#or 
#data.overall[1]
data.loc[1,["player_name"]]
#data.iloc[0]["player_name"]
# Selecting only some columns 
#nationality	position	age
#data[["player_name","nationality","position","age"]]
data.reindex(columns=["player_name","nationality","position","age"]).head(10)
# Slicing and indexing series
data.loc[1:10,"player_name":"league"] 
# reversed sliced data 
data.loc[30:10:-1,"player_name":"nationality"]
# Creating boolean series
boolean = data.height > 190
data[boolean]
# Combining filters
first_filter = data.overall > 90
second_filter = data.age < 22
data[first_filter & second_filter]
data.index.name = "id"
data
# Overwrite index
data3=data.tail(100)
data3.index = range(100,200,1)
data3.head()
dic = {"blood":["A","A","B","B","AB","AB","0","0"],"gender":["F","M","F","M","F","M","F","M"],"rh":[1,0,0,1,0,1,0,1],"age":[15,4,72,65,43,36,18,20]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="blood",columns = "gender",values="rh")
df1 = df.set_index(["blood","gender"])
df1
df1.unstack(level=0)
df2 = df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="age",value_vars=["blood","rh"])

df.groupby("gender").mean()   #that means males have rh(+) more than females (according to our ridiculous dataset) 
df.groupby("rh").age.max() 

df.groupby("blood")[["age","rh"]].min() 