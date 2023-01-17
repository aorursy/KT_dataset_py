import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/googleplaystore.csv')
data.head()
data.columns = data.columns.str.replace(' ', '_')
print("Shape of data (samples, features): ",data.shape)
print("Data Types: \n", data.dtypes.value_counts())
data.Size.value_counts().head()
#please remove head() to get a better understanding 
data.Size=data.Size.str.replace('k','e+3')
data.Size=data.Size.str.replace('M','e+6')
data.Size.head()
def is_convertable(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
    
temp=data.Size.apply(lambda x: is_convertable(x))
temp.head()
data.Size[~temp].value_counts()
data.Size=data.Size.replace('Varies with device',np.nan)
data.Size=data.Size.replace('1,000+',1000)
data.Size=pd.to_numeric(data.Size)
data.hist(column='Size')
plt.xlabel('Size')
plt.ylabel('Frequency')
data.Installs.value_counts()
data.Installs=data.Installs.apply(lambda x: x.strip('+'))
data.Installs=data.Installs.apply(lambda x: x.replace(',',''))
data.Installs=data.Installs.replace('Free',np.nan)
data.Installs.value_counts()
data.Installs.str.isnumeric().sum()
data.Installs=pd.to_numeric(data.Installs)
data.Installs=pd.to_numeric(data.Installs)
data.Installs.hist();
plt.xlabel('No. of Installs')
plt.ylabel('Frequency')
data.Reviews.str.isnumeric().sum()
data[~data.Reviews.str.isnumeric()]
data=data.drop(data.index[10472])
data[10471:].head(2)
data.Reviews=data.Reviews.replace(data.Reviews[~data.Reviews.str.isnumeric()],np.nan)
data.Reviews=pd.to_numeric(data.Reviews)
data.Reviews.hist();
plt.xlabel('No. of Reviews')
plt.ylabel('Frequency')
print("Range: ", data.Rating.min(),"-",data.Rating.max())
data.Rating.dtype
print(data.Rating.isna().sum(),"null values out of", len(data.Rating))
data.Rating.hist();
plt.xlabel('Rating')
plt.ylabel('Frequency')
data.Type.value_counts()
data.Price.unique()
data.Price=data.Price.apply(lambda x: x.strip('$'))
data.Price=pd.to_numeric(data.Price)
data.Price.hist();
plt.xlabel('Price')
plt.ylabel('Frequency')
temp=data.Price.apply(lambda x: True if x>350 else False)
data[temp].head(3)
data.Category.unique()
data.Category.value_counts().plot(kind='bar')
data.Content_Rating.unique()
data.Content_Rating.value_counts().plot(kind='bar')
plt.yscale('log')
data.Genres.unique()
sep = ';'
rest = data.Genres.apply(lambda x: x.split(sep)[0])
data['Pri_Genres']=rest
data.Pri_Genres.head()
rest = data.Genres.apply(lambda x: x.split(sep)[-1])
rest.unique()
data['Sec_Genres']=rest
data.Sec_Genres.head()
grouped = data.groupby(['Pri_Genres','Sec_Genres'])
grouped.size().head(15)
twowaytable = pd.crosstab(index=data["Pri_Genres"],columns=data["Sec_Genres"])
twowaytable.head()
twowaytable.plot(kind="barh", figsize=(15,15),stacked=True);
plt.legend(bbox_to_anchor=(1.0,1.0))
data.Last_Updated.head()
from datetime import datetime,date
temp=pd.to_datetime(data.Last_Updated)
temp.head()
data['Last_Updated_Days'] = temp.apply(lambda x:date.today()-datetime.date(x))
data.Last_Updated_Days.head()
data.Android_Ver.unique()
data['Version_begin']=data.Android_Ver.apply(lambda x:str(x).split(' and ')[0].split(' - ')[0])
data.Version_begin=data.Version_begin.replace('4.4W','4.4')
data['Version_end']=data.Android_Ver.apply(lambda x:str(x).split(' and ')[-1].split(' - ')[-1])
data.Version_begin.unique()
twowaytable = pd.crosstab(index=data.Version_begin,columns=data.Version_end)
twowaytable.head()
twowaytable.plot(kind="barh", figsize=(15,15),stacked=True);
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xscale('log')
data.Version_end.unique()
data.Current_Ver.value_counts().head(6)
data.Current_Ver.isna().sum()
import re
temp=data.Current_Ver.replace(np.nan,'Varies with device')
temp=temp.apply(lambda x: 'Varies with device' if x=='Varies with device'  else  re.findall('^[0-9]\.[0-9]|[\d]|\W*',str(x))[0] )
temp.unique()
data['Current_Ver_updated']=temp
data.Current_Ver_updated.value_counts().plot(kind="barh", figsize=(15,15));
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xscale('log')

