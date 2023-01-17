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
df= pd.read_csv('../input/globalterrorismdb_0718dist.csv', engine="python")
df.info()
df.head()
df.columns
df.country.plot(kind = 'hist',bins = 50,figsize = (12,12),color="red")
plt.show()


filter_tr= (df.country_txt == "Turkey")
df[filter_tr]
df[filter_tr].iyear.plot(kind = 'line', color = 'g',label = 'Year',linewidth=1,grid = True,linestyle = '--',figsize=(15,15))

plt.xlabel('Sum of attacks in Turkey')             
plt.ylabel('Year')
plt.title('Line Plot') 
plt.show()
filter_success = (df.success ==1)
percentage= len(df[filter_success & filter_tr]) / len(df[filter_tr]) *100
print("the percentage of successfull terrorist attacks in Turkey is "+str(percentage))
print(df[filter_tr]['provstate'].value_counts(dropna =False))