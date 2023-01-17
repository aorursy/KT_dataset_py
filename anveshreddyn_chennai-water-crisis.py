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
import numpy as np 

import pandas as pd

import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt 

import os

#Loading Data

level = pd.read_csv("../input/chennai_reservoir_levels.csv")

rain = pd.read_csv("../input/chennai_reservoir_rainfall.csv")

level['Date']= pd.to_datetime(level['Date'])

level.head()


plt.figure(figsize=(20,5))

sns.lineplot(x= 'Date', y= 'POONDI', data= level)
plt.figure(figsize=(20,5))

sns.lineplot(x= 'Date', y= 'CHOLAVARAM', data= level, color= 'G')
plt.figure(figsize=(20,5))

sns.lineplot(x= 'Date', y= 'REDHILLS', data= level,color= 'R')
plt.figure(figsize=(20,5))

sns.lineplot(x= 'Date', y= 'CHEMBARAMBAKKAM', data= level,color= 'Y')
level['total']= level['POONDI']+level['CHOLAVARAM']+level['REDHILLS']+level['CHEMBARAMBAKKAM']

plt.figure(figsize=(20,5))

sns.lineplot(x='Date',y='total',data= level)

rain['Date']=pd.to_datetime(rain['Date'])



rain.head()
sns.set(rc={'figure.figsize':(20,5)})



plt.plot(rain['POONDI'])

plt.plot(rain['CHOLAVARAM'])

plt.plot(rain['CHEMBARAMBAKKAM'])

plt.plot(rain['REDHILLS'])

plt.legend()

plt.title('Rainfall in Reservoir')