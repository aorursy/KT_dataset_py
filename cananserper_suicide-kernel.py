# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#read data and print data headers
data=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
data.head()

data.info()

#corr values
data.corr()

# corr values visualization
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#population and year plot
data.population.plot(kind = 'line', color = 'g',label = 'population',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.year.plot(color = 'b',label = 'year',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')    
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')           
plt.show()
data.iloc[:,1:5].describe()
data.isnull().sum()


#suicide values by gender and age 
ax = sns.catplot(x="sex", y="suicides_no",col='age', data=data, estimator=median,height=4, aspect=.7,kind='bar')
#years plot
data.year.plot(kind = 'hist',bins = 50,figsize = (15,15))
plt.show()
### Scatter Plot 
# x = suicides_no, y = population
##by male and female
## for age
g = sns.FacetGrid(data, row="sex", col="age", margin_titles=True)

g.map(plt.scatter, "suicides_no","population", edgecolor="r")