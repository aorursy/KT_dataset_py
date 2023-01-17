# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')
data
from matplotlib import pyplot as plt
type (data)
type (data.state)
plt.plot(data.state,data.suspects)

plt.show()
plt.plot(data.state,data.deaths)

plt.show()
plt.plot(data.cases,data.deaths)

plt.show()
time=data.date
time
cases=data.cases
cases
plt.plot(data.date,data.cases)

plt.show()
data1=pd.Series((time))
print (data1.agg(['count']))
data2=pd.Series((cases))
print (data2.agg(['max','min','std','max']))
deaths=data.deaths
deaths
deaths=data.deaths

deaths

print (deaths.agg(['max','min','std','max']))
ax=plt.axes()

ax.set_xlabel('cases')

ax.set_ylabel('deaths')

plt.plot(data.cases,data.deaths)

plt.show()
ax=plt.axes()

ax.set_xlabel('time')

ax.set_ylabel('deaths')

plt.plot(data1,deaths)

plt.show()
deaths.plot(kind='pie')
