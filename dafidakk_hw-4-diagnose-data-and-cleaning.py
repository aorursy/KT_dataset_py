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
dataframe=pd.read_csv("../input/air_quality_Nov2017.csv")
dataframe.head() #head shows first 5 rows
# this is first look 
dataframe.tail()
#tail gives last 5 rows
dataframe.columns
#columns gives columns name of the features
dataframe.shape
#gives the shape of the dataset in a tuble (row,columns)
dataframe.info()
#info give fatures names in columns and what type of value in it. Like int,obj,float
# with the quantites of values,we decide are there any NaN values or not.
print(dataframe['PM10 Value'].value_counts(dropna=False))
dataframe.describe()
dataframe.boxplot(column='PM10 Value',by='Longitude')
plt.show()

# in this box plot, at the 2.1239 longitude has a few higher outliers..

 #new data for example from origin our data
    
dataframe_new=dataframe.head()
dataframe_new
melted=pd.melt(frame=dataframe_new,id_vars='Station',value_vars=['Longitude','Latitude'])
#frame=dataframe_new melt edeceğimiz datayı frame e eşitliyoruz
#id_vars= değişmeden kalmasını istediğimiz feature u seçiyoruz
#value_vars='Longitude','Latitude' yeni valuelar olarak oluşturulmasını istediğimiz feature lar
#yeni tablodaki columns name variable ve value default olarak geliyor ileride değiştirilebilir
melted
#PIVOTİNG DATA
#REVERSE OF MELTİNG
melted.pivot(index='Station',columns ='variable',values= 'value')
#row concatenating 
dataframe1=dataframe.head() #dataframe1 created from dataframe's head 
dataframe2=dataframe.tail() #dataframe1 created from dataframe's tail 
concat_data_row=pd.concat([dataframe1,dataframe2],axis=0,ignore_index=True) #pd.concat is the cncatenating method. axis=0 means row concatenating 
concat_data_row   #ignore_index=True new index after the concatenating
#columns concatenating 
dataframe1=dataframe['Longitude'].head() #dataframe1 created from dataframe['Longitude']'s  head 
dataframe2=dataframe['Latitude'].head() #dataframe1 created from dataframe['Latitude']'s head 
concat_data_columns=pd.concat([dataframe1,dataframe2],axis=1)  # axis=1 means concatenate columns side by side
concat_data_columns
dataframe.dtypes
#some data types are convertable for example:

dataframe["Station"]=dataframe["Station"].astype("category")
dataframe.dtypes
dataframe.info()
dataframe["O3 Quality"].value_counts(dropna=False)
# 1476 NaN value and 167 -- value in this column.
#droping Nan Values
dataframe3=dataframe
dataframe3["O3 Quality"].dropna(inplace=True)
dataframe3["O3 Quality"].value_counts(dropna=False)

assert dataframe3["O3 Quality"].notnull().all()
# assert statment  check rest of the code is true or not, when is true return nothing but when its wrong traceback errors come.
#like assert 1==2 check this statment
dataframe4=dataframe
dataframe4["O3 Quality"].fillna('empty',inplace=True)
assert dataframe4["O3 Quality"].notnull().all()
dataframe4["O3 Quality"].value_counts(dropna=False)

#assert examples

assert dataframe.columns[0] == 'Station' #nothing returns so its correct
assert dataframe.Longitude.dtypes==np.float # nothing returns so its correct


#note: indentation error means line stars with empty space!!