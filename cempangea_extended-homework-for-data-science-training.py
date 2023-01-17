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
data = pd.read_csv('../input/forestfires.csv')
data.head(20) #show to only 20 rows
data.describe(include='all')
data.columns #show to features
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.RH.plot(kind = 'line', color = 'g',label = 'RH',figsize = (7,7),linewidth=1,alpha = 0.9,grid = True,linestyle = ':') 
data.temp.plot(color = 'r',label = 'temp',linewidth=1,figsize = (6,6), alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Days that taken measurement')              # label = name of label
plt.ylabel('Frequency')
plt.title('RH-temp Line Plot')            # title = title of plot
plt.show()
data.plot(subplots = True, figsize = (16,16))
plt.xlabel('Days that taken measurement') 
plt.ylabel('Frequencies')
plt.show()
data.wind.plot(kind = 'hist',bins = 50,figsize = (12,12),range=(0,8),grid = True,normed = True)  #bins defines number of bar; normed=normalization
plt.show()
data.wind.plot(kind = 'hist',bins = 10000,figsize = (12,12),range=(0,8),normed = True,cumulative = True)  
plt.show()
x = data['temp'] >= 30
data[x]
series = data['ISI']        # data['Defense'] = series
print(type(series))
data_frame = data[['ISI']]  # data[['Defense']] = data frame
print(type(data_frame))
# 1 - Filtering Pandas data frame
x = data['rain']>=0.5     # There are only 3 pokemons who have higher defense value than 200
data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['rain']>=0) & (data['temp'] >= 30)]
data.wind[data.temp>=30]
data.shape
print(data['ISI'].value_counts(dropna =False))    #dropna=False = to show Nan 
Mean_temp = sum(data.temp)/len(data.temp)
data["temp_level"] = ["high" if i > Mean_temp else "low" for i in data.temp]
data.loc[:7,["temp_level","temp"]]
data.boxplot(column='temp')
data_new = data.head()   
data_new
melted = pd.melt(frame=data_new,id_vars = 'month', value_vars= ['FFMC','DC'])
melted
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = data['month'].head()
data2= data['temp'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # combine to columns
conc_data_col
data.dtypes
data['RH '] = data['RH'].astype('float64')
data.dtypes
data.info()
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2
data2.resample("A").mean()  #A : year M: mount
data2.resample("M").mean()  #A : year M: mount 
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
def div(n):
    return n**2
data2.temp.apply(div)
data1 = data.set_index(["temp","wind"]) 
data1.head(100)
data.groupby("rain").mean()

# indexing using square brackets
data["temp"][1]
# using loc accessor
data.loc[1,["temp"]]
# Selecting only some columns
data[["temp","rain"]]
# Slicing and indexing series
data.loc[1:10,"temp":"rain"]   
# From something to end
data.loc[1:10,"ISI":] 
# Creating boolean series
boolean = data.temp > 30
data[boolean]
# Combining filters
first_filter = data.temp > 30
second_filter = data.ISI > 10
data[first_filter & second_filter]
data.head()
# our index name is this:
print(data.index.name)
# lets change it

data.index.name = "index_name"
data.head()
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["month","temp"]) 
data1.head(100)
