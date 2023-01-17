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
# to upload csv data:
data =pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
# to get general info about the csv file:
data.info()
# to see the columns of the csv file:
data.columns
# in order to get an idea about the data, show first 7 rows of the csv file:
data.head(7)
# loading csv again with only the columns I want to use:
ndata = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1',usecols=[0, 1, 2, 3, 7, 8, 12, 13, 14, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 83, 84, 98, 101])
# showing columns:
ndata.columns
# filtering data to show terrorism in Turkey:
turdata = ndata[ndata.country_txt == "Turkey"]
turdata.head()
# showing the correlation between the columns:
turdata.corr()
# visualization of data.corr():
f,ax = plt.subplots(figsize=(13, 13))
sns.heatmap(turdata.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# filtering terrorist attacks againts military targets:
mildata = turdata[turdata.targtype1 == 4]
mildata.head()
# showing lineplot of nu of terrorist attacks against military targets throug years:
mildata.iyear.plot(kind = 'line', color = 'r', label = 'Nu of Terror Atacks', linewidth = 2, alpha = 0.8, grid = True, linestyle = ':', figsize = (17,5))
plt.legend(loc = 'lower right')
plt.xlabel('Number of Terror Attacks')
plt.ylabel('Years')
plt.title('Nu of Terror Attacks to Turkish Military Aspects')
plt.show()
# Showing the terror attack types:
plt.hist(mildata.attacktype1, bins = 8)
plt.xlabel('Terror Attack Types')
plt.ylabel('Number of Terror Attacks')
plt.show()
print(mildata.attacktype1.unique())
print(mildata.attacktype1_txt.unique())
plt.scatter(mildata.longitude, mildata.nkill, color = 'r', alpha = 0.3, label = 'Nu of Deaths')
plt.scatter(mildata.longitude, mildata.nwound, color = 'b', alpha = 0.3, label = 'Nu of Wounds')
plt.xlabel('Longitudes (West to East)')
plt.ylabel('Number of Casualties')
plt.title('Deaths from Terror Attacks According to Longitude')
plt.legend(loc = 'upper left corner')
plt.show()
# (filtering example) Number of terror attacks by bombs concluded with deaths:
bomb_data = mildata[(mildata['nkill']>=1) & (mildata['attacktype1']==3)]

# Or we can use this code:
#bomb_data = mildata[np.logical_and(mildata['nkill']>=1, mildata['attacktype1']==3)]

print('Number of Terror Attacks Using Bombs Concluded With Deaths:', bomb_data['nkill'].count())
print('Total Number of Deaths:', int(bomb_data['nkill'].sum()))