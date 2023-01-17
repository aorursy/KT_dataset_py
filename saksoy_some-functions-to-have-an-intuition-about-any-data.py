# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting Library for data
import seaborn as sns # Plotting Library for data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# To use any kind of data, first we should load the data. Since the current data in .csv format we use pandas read_csv method.
## Important note: Because the default utf-8 can't decode some bytes and throws an error, encoding parameter with a value ISO-8859-1 is added.
## Changing the default engine parameter may be the other solution for this error
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
#check the data loaded properly
data.head() #show first 5 records with headers
#Rename column names with more meaningful text
data.rename(columns={'iyear':'Year',
                     'imonth':'Month',
                     'iday':'Day',
                     'approxdate':'ApproxDate',
                     'country':'CountryCode',
                     'country_txt':'Country',
                     'region':'RegionCode',
                     'region_txt':'Region',
                     'attacktype1_txt':'AttackType',
                     'target1':'Target',
                     'targtype1_txt':'TargetType',
                     'weaptype1_txt':'WeaponType',
                     'attacktype1':'AttackTypeCode',
                     'attacktype2':'AttackTypeCode2',
                     'attacktype2_txt':'AttackType2',
                     'attacktype3':'AttackTypeCode3',
                     'attacktype3_txt':'AttackType3',
                     'targtype1':'TargetTypeCode',
                     'targsubtype1':'TargetSubtypeCode',
                     'weapsubtype4':'WeapSubtypeCode',
                     'weapsubtype4_txt':'WeapSubtype',
                     'propextent':'PropExtendCode',
                     'propextent_txt':'PropExtend'},inplace=True)
#have a look at the data with new column names
data.head()
#General info about data columns
data.info(1)
#Result: the dataframe has 135 column with different types (int64, object, float64) and has 170350 records in total 
#basic statistical information of columns
data.describe()
## correlation between columns
##This is a matrix version of data, and shows if any meaningful relations between columns.
## +1 maximum positive correlation
## -1 maximum negative correlation
## 0 no correlation at all
## e.g: released and claimmode3 have one the most powerful positive correlation with the score of 068
## non-numeric columns automatically eliminated by python
data.corr()
#Correlation Graph with Seaborn

_,ax = plt.subplots(figsize=(50, 50))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot with Matplotlib
# for selecting data: data.TableColumn.plot--->kind=line for line plot
# legend=legend box of data columns
# title=graph title
# tick_params=to change the font size of labels
# plt.show=for drawing plot without an output with an extra text information

dataX=data[0:500] #first 500 records of the original dataset
dataX.TargetSubtypeCode.plot(kind = 'line', color = 'green',label = 'TargetSubtypeCode',linewidth=1,alpha = 0.8,grid = True,linestyle = '-')
dataX.TargetTypeCode.plot(kind = 'line', color = 'red',label = 'TargetTypeCode',linewidth=1.5,alpha = 0.8,grid = True,linestyle = '-',figsize = (60,12))
plt.legend(loc='upper right')
plt.title('Line Plot',fontsize=40)    
plt.legend(loc='upper right',prop={'size': 32})
plt.tick_params(axis='both', which='major', labelsize=30)
plt.show()
#Histogram  with Matplotlib
# for selecting data: data.TableColumn.plot--->kind=hist for histogram plot
dataX=data[0:500] #first 500 records of the original dataset
dataX.TargetSubtypeCode.plot(kind = 'hist', color = 'green',label = 'TargetSubtypeCode',linewidth=1,bins = 30,alpha = 0.5,grid = True)
dataX.TargetTypeCode.plot(kind = 'hist', color = 'red',label = 'TargetTypeCode',bins = 20,linewidth=1,alpha = 0.8,grid = True,figsize = (60,12))
plt.legend(loc='upper right',prop={'size': 32})
plt.tick_params(axis='both', which='major', labelsize=30)
# Scatter with Matplotlib
# for selecting data: data.plot--->kind=scatter for scatter plot, x=data column for x axis, y=data column for y axis
data.plot(kind='scatter', x='TargetSubtypeCode', y='TargetTypeCode',alpha = 0.8,color = 'red')
plt.xlabel('TargetSubtypeCode')              
plt.ylabel('TargetTypeCode')
plt.title('Scatter Plot')            