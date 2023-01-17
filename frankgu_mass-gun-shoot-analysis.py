# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import folium

import pandas as pd

import numpy as np

from numpy import math

import matplotlib.pyplot as plt

import seaborn as sns
#load data and see the structure of data to clear how to analysis data

data = pd.read_csv('../input/Mass Shootings Dataset.csv',encoding = "ISO-8859-1")
data.head()
data.info()
map_osm = folium.Map(location = [39,-98])

map_osm
lt = data.Latitude

lg = data.Longitude
for a,b in zip(lt,lg):

    if math.isnan(a) == False:

        folium.Marker([a,b]).add_to(map_osm)
map_osm
#in order to groupby the data by years, extract year information from Date column then group it

#I use total victims to draw the plot to show how many people got killed and injured per year





data['year'] = pd.to_datetime(data.Date).dt.year

datasum = data.groupby('year').sum()

datasum['year'] = datasum.index

datasum.index = range(1,len(datasum)+1)
datasum
plt.figure(figsize=(16,8))

plt.bar(list(datasum['year']),list(datasum['Total victims']))

plt.title('Total Victims Per Year',fontsize = 24)

plt.xticks(datasum.year, rotation='vertical',fontsize = 12)

plt.yticks(fontsize = 12)
plt.show()
#Because the race and gender data have redundant name, so I replace the name to make it more explicitly

#Then use heatmap to show the correlation for gender and race



data['Race'].value_counts()
data.Gender.value_counts()
data.Gender.replace(['M','M/F','Male/Female'],['Male','Male/Female','Unknown'],inplace= True)
data.Gender.value_counts()
data.Race.unique()
data.Race.replace(['white','White ','black','Other','unclear'],['White','White','Black','Unknown','Unknown'],inplace=True)
data['Race'].value_counts()
crosstab_gender = pd.crosstab(data.Race,data.Gender)

crosstab_gender
crosstab_race = pd.crosstab(data.Gender,data.Race)
sns.heatmap(crosstab_gender.corr(), annot=True, linewidths=.5)
plt.show()
sns.heatmap(crosstab_race.corr(), annot=True, linewidths=.5)
plt.show()
x = pd.to_datetime(data.Date).dt.month.value_counts()
df = pd.DataFrame(x).sort_index()

df.plot(kind = 'bar',rot = 0,colormap = 'Accent',fontsize = 12)

plt.title('Total gunslinging for every month',fontsize = 18)
plt.show()
data['Mental Health Issues'].replace(['Unclear ','unknown'],['Unclear','Unknown'],inplace = True)

count_mental = data.groupby('Mental Health Issues').count().Title

count_mental
count_mental.plot(kind = 'bar',rot = 0,fontsize =12)

plt.title('Mental condition with shooter',fontsize = 18)

plt.show()