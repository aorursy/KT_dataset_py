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
import pandas as pd

import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt



import pandas as pd

IncData= pd.read_csv("../input/inc5000/Data Set- Inc5000 Company List_2014.csv")


IncData.plot(x="rank",y="growth")



#The bigger the growth the higher the ranking which looks right.


hist=IncData.hist(column="yrs_on_list")



#looks like it is the first year on the list for most companies 


x=IncData[["rank"]]

y=IncData[["revenue"]]

IncData.plot(kind="scatter",x="rank",y="revenue")



#looks like most companies has small revenues and ranking does not reflect the size of revenue

size=IncData[['workers']]

IncData[['rank','revenue']].plot(kind='scatter',x='rank',y='revenue', s=size, alpha=.1 )



#result: No, in fact the more workers a company has the slower the growth
IncData[['workers','revenue']].plot(kind='scatter',x='workers',y='revenue', s=50, alpha=.2 )
Health= IncData.loc[IncData['industry']=='Health'].count()[1]

NonHealth=IncData.loc[IncData['industry']!='Health'].count()[1]



colors =['LightSalmon',"PaleTurquoise"]

labels = 'Health', 'NonHealth'

sizes = [Health, NonHealth]

explode = (0, 0.1) 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        colors=colors, shadow=False, startangle=90)

ax1.axis('equal')  



plt.show()
