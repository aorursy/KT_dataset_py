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

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

from collections import Counter

import calendar

import re

import pylab as plt

import numpy as np



df = pd.read_csv('../input/independence-days.csv')

years = df['Year celebrated']

years_count = years.value_counts()

years_count[0:9].reset_index().plot(x='index', y='Year celebrated', kind = 'bar')
date = df['Date of holiday'].str.split(' ').str[0]

date_count = date.value_counts()

date_count[0:9].reset_index().plot(x='index', y='Date of holiday', kind = 'bar')

date_1=[datetime.strptime(x, '%m-%d').date() for x in date]
month = [x.month for x in date_1]

month_count=Counter(month)

A= list(month_count.keys())

month_name = []

for i in range(0,len(month_count)):

   month_name.append(calendar.month_abbr[A[i]])

x_pos = np.arange(len(A))

Y = [month_count[k] for k in A]

plt.bar(x_pos,Y, align='center')

plt.xticks(x_pos, month_name)

plt.show()   





day = [x.day for x in date_1]

day_count=Counter(day)



B= list(day_count.keys())



x_pos = np.arange(len(B))

Y = [day_count[k] for k in B]

plt.bar(B,Y, align='center')

plt.show()   

Independence_From = [s for s in df['Event celebrated'] if "Independence from" in s]

From_country = [s for s in df['Event celebrated'] if "Independence from" in s]

From_country_1 = []

i=0

for s in From_country:

    From_country_1.append(re.findall(r'from(.*?) in',s))

    

From_country = pd.DataFrame(From_country_1, columns=['country'])

From_country_count=From_country['country'].value_counts()



From_country_count[0:6].reset_index().plot(x='index', y='country',kind='bar',rot=90)