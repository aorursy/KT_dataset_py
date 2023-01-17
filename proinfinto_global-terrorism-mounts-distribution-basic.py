# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Bacis visualization tool.
import seaborn as sns # Cool visualization tools. ^^


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='Windows-1252')

#11.11.2018 
#pd.read_csv('../input/globalterrorismdb_0718dist.csv') 
#I used the above code because I received an error.(utf-8 Error!)


data.head(3)
data.info()
data.columns
globaldata = data[['iyear','imonth','country','country_txt','region','region_txt','city','success','attacktype1','attacktype1_txt']].copy()
type(globaldata)
globaldata.head(10)
January =  globaldata['imonth']<2
globaldata[January].head(5)
#Nice just January..
globaldata.imonth.plot(kind = 'hist',bins = 50, figsize = (15,15))
plt.show()

# so how do all months look?
globaldata.iyear.plot(kind = 'hist',bins = 100, figsize = (15,15))
plt.show()
dropzero = globaldata['imonth'] != 0
globaldata[dropzero].head(5)
globaldata[dropzero].describe()
year_count = globaldata[dropzero].groupby(['iyear']).count()
month_count = globaldata[dropzero].groupby(['imonth']).count()


plt.plot(month_count,color='red',label = "Monthly Distribution")
plt.title('Monthly Distribution')
plt.xlabel('Mounts')
plt.ylabel('count')
plt.show()

plt.plot(year_count,color='red',label = "Years Distribution")
plt.title('Years Distribution')
plt.xlabel('Year')
plt.ylabel('count')
plt.show()
