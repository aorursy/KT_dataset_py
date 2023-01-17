# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
data_path='../input/US GDP.csv'
us_gdp = pd.read_csv(data_path)
us_gdp
us_gdp.head()
us_gdp.tail()
us_gdp.iloc[33]
us_gdp.loc[30]
us_gdp.iloc[32:34]
us_gdp.plot(kind='scatter',x='Year',y='US_GDP_BN',title='US GDP Per Year')
us_gdp.plot(kind='scatter',x='Year',y='US_GDP_BN',title='US GDP Per Year')
plt.title("US GDP per Year")
plt.ylabel("GDP")
plt.suptitle("From %d to %d" %())
us_gdp.Year.astype('str')
help(min)
us_gdp.Year.min()
us_gdp.plot(kind='scatter',x='Year',y='US_GDP_BN',title='US GDP Per Year')
plt.suptitle("US GDP per Year")
plt.ylabel("GDP")
plt.title("From %d to %d" %(us_gdp['Year'].min(),
                              us_gdp['Year'].max()))
axes = us_gdp.plot(kind='line',x='Year',y='US_GDP_BN',title='US GDP Per Year',figsize= (12,5) )
us_gdp.plot(kind='line',x='Year',y='US_GDP_BN',title='US GDP Per Year',ax=axes)
plt.suptitle("US GDP per Year", size=15)
plt.ylabel("GDP")
plt.title("From %d to %d" %(us_gdp['Year'].min(),
                              us_gdp['Year'].max()))
us_gdp.plot(kind='line',x='Year',y='US_GDP_BN',title='US GDP Per Year',ax=axes)
plt.suptitle("US GDP per Year", size=15)
plt.ylabel("GDP")
plt.title("From %d to %d" %(us_gdp['Year'].min(),
                              us_gdp['Year'].max()))
us_gdp.plot(kind='bar',x='Year')

us_gdp['GDP_Growth_PC'].plot(kind='bar')
