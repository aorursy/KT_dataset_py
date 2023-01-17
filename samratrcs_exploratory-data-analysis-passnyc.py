# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns





# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
kaggledf = pd.read_csv("../input/2016 School Explorer.csv")


# Any results you write to the current directory are saved as output.
#########Find number of school city wise
#####Find Unique values in city
cUnique = kaggledf.City.unique()
sizeCitydf = pd.DataFrame(kaggledf.groupby("City").size(), columns = ['count'])
sizeCitydf.reset_index(inplace = True)
axcity = sns.barplot(x='City', y = 'count', data = sizeCitydf)
axcity = axcity.set_xticklabels(axcity.get_xticklabels(), rotation = 90)
#########Number of community school city wise
kaggledf['Community School?'].isnull().any()
sizeCiCodf = pd.DataFrame(kaggledf.groupby(["City","Community School?"]).size(), columns = ['count'])
sizeCiCodf.reset_index(inplace = True)
sizeCiCodf = sizeCiCodf.pivot(index='City', columns='Community School?')['count']
sizeCiCodf.dropna(inplace = True)
sizeCiCodf.reset_index(inplace = True)
sizeCiCodf = pd.melt(sizeCiCodf, id_vars="City", var_name="Community", value_name="cValue")
axcityco = sns.barplot(x='City',y='cValue', hue = 'Community', data = sizeCiCodf)
axcityco = axcityco.set_xticklabels(axcityco.get_xticklabels(), rotation = 90)
########Community school Asian Distribution
ethinicityCol = 'Percent Asian'
ascodstdf = kaggledf[kaggledf['Community School?'] == 'Yes'][ethinicityCol]
ascodstdf = ascodstdf.str.split('%').str[0]
ascodstdf = ascodstdf.astype(int)
ascodstdf = ascodstdf/100
ethdist = sns.distplot(ascodstdf)
ethdist.set(ylabel = "Percent")
####### Absent vs environment 
absenvdf = kaggledf[['Percent of Students Chronically Absent','Supportive Environment %']]
absenvdf['Percent of Students Chronically Absent'] = absenvdf['Percent of Students Chronically Absent'].str.split('%').str[0]
absenvdf['Supportive Environment %'] = absenvdf['Supportive Environment %'].str.split('%').str[0]
absenvdf.dropna(inplace = True)
absenvdf = absenvdf.astype(int)
sns.regplot(x = 'Percent of Students Chronically Absent', y ='Supportive Environment %', data = absenvdf)

####### 
absratdf = kaggledf[['Percent of Students Chronically Absent','Supportive Environment Rating']]
absratdf.dropna(inplace = True)
#print(absratdf['Supportive Environment Rating'].unique())
absratdf['Percent of Students Chronically Absent'] = absratdf['Percent of Students Chronically Absent'].str.split('%').str[0]
absratdf['Percent of Students Chronically Absent'] = absratdf['Percent of Students Chronically Absent'].astype(int)
column = 'Percent of Students Chronically Absent' 
absratdf[column] = np.where((absratdf[column]> 0 )&(absratdf[column] <= 25 ) , 1, 
        np.where((absratdf[column] > 25 )&(absratdf[column] <= 50), 2, 
                 np.where((absratdf[column] > 50 )&(absratdf[column] <= 75), 3, 4)))

#print(absratdf.columns.values)
#print(absratdf.head(5))

print(pd.crosstab(absratdf[column], absratdf['Supportive Environment Rating'],margins=True ))
