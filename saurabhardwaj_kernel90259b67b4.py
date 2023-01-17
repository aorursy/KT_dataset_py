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
os.getcwd()
os.chdir("/kaggle/input/india-trade-data")
imported = pd.read_csv("2018-2010_import.csv")
imported.head()
exp = pd.read_csv("2018-2010_export.csv")
exp.head()
imported.shape

exp.shape
group = imported.groupby('country').sum()[['value']]
# gives the countries from which we import at most

group.sort_values(['value'],axis=0,ascending=False)
expgroup = exp.groupby('country').sum()[['value']]
# countries in which we export at most

expgroup.sort_values('value',axis = 0,ascending = False)
#from above we can see that in which country how much we import and export & also see top import and export countries,
# now we see that which sector import and export in other countrise

impsector = imported.groupby('country').count()[['Commodity']]
print(impsector)
impsector1 = imported.groupby('Commodity').count()[['country']]
print(impsector1)


impsector2 = imported.groupby('Commodity').sum()[['value']]
print(impsector2)
# shows highest imported sectors

impsector2.sort_values('value',axis = 0,ascending = False)
imp1 = imported.groupby(['year','country']).sum()['value']
print(imp1)
imp1.shape
imp1.head()
# each sector's share in importing 

imp2 = pd.pivot_table(imported,'value',['country','year','Commodity'])
print(imp2)
imp3 = imported.sort_values('value',ascending = False,axis = 0)
imp3.head()
pd.pivot_table(imp3,'value',['country','year','Commodity'])
pd.pivot_table(imported,'value',['year','country'])
all_import = imported.set_index(['year','country','value']).sort_index()
import matplotlib.pyplot as plt
# graph shows the highest import in year

plt.plot(imported['year'],imported['value'])

plt.xlabel("year") 

plt.ylabel("value") 

plt.show
pd.pivot_table(exp,'value',['year','country'])
pd.pivot_table(imported,'value',['year','Commodity','country'])
exp1 = exp.groupby('Commodity').sum()[['value']]
print(exp1)
# in which we export at most

exp1.sort_values('value',ascending = False,axis = 0)
# graph shows in which sector we export at most in particular year

plt.plot(exp['year'],exp['value'])

plt.xlabel('year')

plt.ylabel('value')

plt.show()