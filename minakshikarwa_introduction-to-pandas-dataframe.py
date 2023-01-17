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
weather = {'city' : ['Mumbai','Delhi','Gurgaon','Jaipur'],

           'temperature' : [30, 40, 43, 38 ],

           'wind_speed'  : [15, 25, 30, 10]

}

weather
df = pd.DataFrame(weather)

df
df.iloc[:,1]
df.loc[:,'temperature']
df.temperature

# it also can be written as df['temperature']
temp = np.random.randint(low=20,high=50,size=[10,])

city = np.random.choice(['Mumbai','Delhi','Chennai','Kolkata'],10)
mylist = list(zip(temp,city))

mylist
df1 = pd.DataFrame(mylist,columns=['temp','city'])

df1