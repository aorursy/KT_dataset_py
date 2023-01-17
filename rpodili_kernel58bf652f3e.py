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

import numpy as np

Jounery = pd.read_csv("../input/00DAC437-FF8B-4DA3-9E24-4EE1B1AA12EC.csv")





df = pd.DataFrame()

for i in np.arange(len(Jounery)):

    avg_x_value = np.mean(Jounery.iloc[i+1:i+11,8])

    avg_y_value = np.mean(Jounery.iloc[i+1:i+11,9])

    avg_z_value = np.mean(Jounery.iloc[i+1:i+11,10])

    Jounery.iloc[i,8] = avg_x_value

    Jounery.iloc[i,9] = avg_y_value

    Jounery.iloc[i,10] = avg_z_value

    i= 1+10

    

    
# Jounery.iloc[0,8]



df = Jounery[Jounery['type']=='gps']



# print(df[df['speed']==0 and df['x']==0.0])



# df[' timestamp '] = pd.to_datetime(df[' timestamp ']).apply(lambda x: pd.datetime.fromtimestamp(x).date())

df[' timestamp '] = pd.to_datetime(df[' timestamp '], unit='ms')



print(df.head(5))

# print(pd.to_datetime(1434838676097, unit='ms'))



import matplotlib.pyplot as plt



plt.scatter(df['speed'], df[' timestamp '])

plt.title('Scatter plot')

plt.xlabel('x')

plt.ylabel('y')

plt.ylim("2015-06-20 22:17", "2015-06-20 22:25")

plt.show()
accidents = df[df['speed']==0]

print(accidents.head(5))

print('Total number of incidents consider as accidents are',accidents.size)
plt.scatter(df['bearing'], df['x'])
plt.scatter(df['bearing'], df['y'])
plt.scatter(df['bearing'], df['z'])