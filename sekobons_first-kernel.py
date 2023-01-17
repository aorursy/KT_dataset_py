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
data=pd.read_csv('../input/vgsales.csv')
data.head(10)
data.columns
data.info()
data.corr()
data.plot(kind='scatter',x='EU_Sales',y='Global_Sales',alpha=0.5,color='blue')
plt.xlabel("NA Sales-") ## label of x
plt.ylabel("Global Sales-")  ## label of y                            
plt.show()           ## EU Sales-Global Sales Correlation

data.NA_Sales.plot(color='g',label='NA Sales',linewidth=2,alpha=0.5,grid=True,linestyle=":")
data.Global_Sales.plot(color='r',label='Global Sales',linewidth=2,alpha=0.5,grid=True,linestyle="-.")
plt.legend()
plt.xlabel("X") 
plt.ylabel("Y")       ## What should I do ? Bad data istatistices. learn the purpose :) 
plt.show()
data[data['Publisher']=="Nintendo"].head(10)
print("Global -- ",data.Global_Sales.mean(),"NA --",data.NA_Sales.mean())
## Na Sales is almost half of Global sales

filter1=data.Year>2016
data[filter1]
## Sales very low after 2016 year
filter1=data.Global_Sales>30             ## very few data in global sales greater than 30
data[filter1]
data.Year.plot(kind='hist',bins=40,grid=True,label='Years')
plt.legend()                                        ## Distribution of years
plt.show()
for index,value in data[["Global_Sales"]][0:2].iterrows():
    print(index,":",value)
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)
plt.show()
data['H_or_S']=["High Sales" if i>20  else "Low Sales" for i in data.Global_Sales]
data["Old_or_New"]=["New" if i>2012 else "Old" for i in data.Year] ## List comprehension
data.head(10) 
data_new=data.head()
melted=pd.melt(frame=data_new,id_vars='Name',value_vars=["NA_Sales","Global_Sales"])
melted           ##First 10 sample group by Name with NA and Global Sales
dataconcat=pd.concat([data.head(),data.tail()],axis=0,ignore_index=True) 
dataconcat                ## vertical concat(axis=0)

dataconcat=pd.concat([data.NA_Sales.head(),data.EU_Sales.head()],axis=1)
dataconcat        ### Na Sales and Eu Sales horizontal concat (axis=1)
data.dtypes
data["Publisher"]=data["Publisher"].astype('category') 
data.dtypes   ## convert object to category data type

data.info()
data["Year"].value_counts(dropna=False) ## Numbers of years (dropna=False) NaN is showed (271)
data2=data.copy()
data2["Publisher"].dropna(inplace=True) ## Dropped NaN values of Publisher
assert data2["Publisher"].notnull().all() ## No error so we dropped nan values of publisher 
data3=data.loc[:,["NA_Sales","Global_Sales","EU_Sales"]] ## Na,Global,Eu Sales in subplots
data3.plot(subplots=True,grid=True,figsize=(10,10))
plt.show()
data.Year.fillna('1970',inplace=True) # if Year is NaN. Fill 1970
time_list=list(data.Year)        
for i in range(0,data.Year.count()):    # Year  converted to int 
    time_list[i]=int(time_list[i])      # because float values can not be converted to datetimes 
    time_list[i]=str(time_list[i])

newdate=pd.to_datetime(time_list)     ## timelist converted to datetimes
data["Date"]=newdate
data2=data.head()
data2=data2.set_index("Date")
data2
data2.resample("A").mean().interpolate('linear') # Average over the  years with interpolate linear
data=data.set_index("Rank") ## index rank was made
data.head()
data[["NA_Sales","Global_Sales"]].head()
print(type(data[["Publisher"]]))  ## dataframe
print(type(data["Publisher"]))    ## series
## Names of games with Global Sales greater than 30
data.Name[data.Global_Sales>30] 
print(data.shape)           # tuple,column
print(data.index.name)      # name of data index 
data2=data.copy()
data2.index=range(100,16698,1) ## index start 100 end 16698
data2.head()
data3=pd.read_csv('../input/vgsales.csv')
data3=data3.set_index(["Publisher","Genre"]) #Publisher and Genre indexing
data3.head(100)
 # averages Global Sales and Na Sales of genres
data3.groupby("Genre")["Global_Sales","NA_Sales"].mean()
# max Global Sales of genres
data.groupby("Genre").Global_Sales.max() 