# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

elevation=pd.read_csv('/kaggle/input/covid-19-usa-elevationcsv/salida_USA.csv')
elevation.head()
elevation['Elevation']=(elevation['elevation']/250).apply(int)
#covid_19['Elevation'].value_counts()
covid_19=elevation[elevation['elevation']>=0]
#covid_19_2=covid_19_2[covid_19_2['estimated_population']>=5000]
covid_19=elevation[elevation['cases']>=100]

a=covid_19[['deaths','cases','Elevation']].groupby(['Elevation']).sum().reset_index()
a['death_rate']=a['deaths']/a['cases']
print(a.shape)
a.head(10)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#ax = sns.scatterplot(x=, y=,hue='Country', data=a)
from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot(a['Elevation'], a['death_rate'], kind="reg", stat_func=r2)
covid_19_2=elevation[elevation.cases>100]
covid_19_2=covid_19_2[covid_19_2.estimated_population<50000]

#a=covid_19_2[['deaths','cases','elevation','county']].groupby(['elevation','county']).sum().reset_index()
covid_19_2['death_rate']=covid_19_2['deaths']/covid_19_2['cases']
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#ax = sns.scatterplot(x=, y=,hue='Country', data=a)
from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot(covid_19_2['elevation'], covid_19_2['death_rate'], kind="reg", stat_func=r2)