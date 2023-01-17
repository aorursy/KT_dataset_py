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
## Importing various modules



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
census2001 = pd.read_csv('../input/all.csv')

census2001.shape
census2001.head()
## Getting the Unique Columns values



census2001.columns
census2001.dtypes
## Checking the unique values of States



census2001.State.unique()
## Retrieving the data only for the state of Maharashtra



mh = census2001[census2001['State'] == 'Maharashtra']
##Checking whether we have got the correct data



mh.head()
## Checking the Dimensions for Maharashtra State data



mh.shape
## Names of all districts in Maharashtra?



mh.District.unique()



## We have 35 total districts in the state of Maharashtra
## Visualizing the population in all the districts





dist_population = mh[['Persons','District']].sort_values(['Persons'],ascending = False)

dist_population
dist_population.plot(kind = "bar",color = 'orange')

plt.xlabel('District')

plt.ylabel('Number of Persons')

plt.title('District vs Number of Persons')
## Number of Literate Persons in each District



dist_literacy = mh[['Persons..literate','District']].sort_values(['Persons..literate'],ascending = False)

dist_literacy.plot(kind = "bar",color = 'blue')

plt.xlabel('District')

plt.ylabel('Literate People')

plt.title('District vs Literate People')
## Number of Women per 1000 men per district



women = mh[['Sex.ratio..females.per.1000.males.','District']].sort_values(['Sex.ratio..females.per.1000.males.'],ascending = False)

women.plot(kind = "bar",color = 'red')

plt.xlabel('District')

plt.ylabel('Women per thousand men')

plt.title('District vs Women per thousand men')

plt.ylim(700,1200)
## How many people have permanent houses



permanent_houses = mh[['Permanent.House','District']].sort_values(['Permanent.House'],ascending = False)

permanent_houses.plot(kind = 'bar',color = 'yellow')

plt.xlabel('District')

plt.ylabel('Permanent Houses')

plt.title('Permanent Houses vs District')