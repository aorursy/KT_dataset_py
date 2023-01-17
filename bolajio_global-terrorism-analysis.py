# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
terrorism=pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1') 
terrorism.head(9)
#Describe data
print(terrorism.describe());

#terrorist attacks over the years
plt.subplots(figsize=(15,5))
sns.countplot('iyear',data=terrorism)
plt.xticks(rotation=90)
plt.title('Terrorist attacks over the Years')
plt.show()


#The spike in terrorist activities between 2012 and 2016, which countries saw the most attacks
terrorism = terrorism[terrorism['iyear'] > 2011]
plt.subplots(figsize=(30,15))
sns.countplot('country_txt',data=terrorism)
plt.xticks(rotation=90)
plt.title('terrorism by country between 2012 and 2016')
plt.show()


#Which regions have seen the most terror attacks 
terrorism = terrorism[terrorism['iyear'] >= 1970]
terror_region=pd.crosstab(terrorism.iyear,terrorism.region_txt)
terror_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
#Has it always been these regions?
terror_region=pd.crosstab(terrorism.iyear,terrorism.region_txt)
terror_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
