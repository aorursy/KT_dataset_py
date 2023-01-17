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
df = pd.read_csv('/kaggle/input/rainfall-data-india-since-1901/rainfall_India_2017.csv')

df.head()
print(df.SUBDIVISION.unique())

print(df.SUBDIVISION.nunique())

print()

print(df.YEAR.unique())

print(df.YEAR.nunique())
df.iloc[:,2:14]
boolean = []

for year in df.YEAR:

    if year == 1901:

        boolean.append(True)

    else:

        boolean.append(False)

is_1901 = pd.Series(boolean)

df_1901 = df[is_1901]
import matplotlib.pyplot as plt

 

x  = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

itrate = len(df_1901.index)

for i in range(itrate):

    plt.plot(x, df_1901.iloc[i,2:14], label=df_1901.iloc[i,0])

    #plt.plot(x, df.iloc[112,2:14], label=df.iloc[112,0])

    plt.plot()



plt.xlabel("Months")

plt.ylabel("Rain data")

plt.title("Line Graph of rainfall in 1901")

plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

plt.show()
for y in df.YEAR.unique():

    boolean = []

    for year in df.YEAR:

        if year == y:

            boolean.append(True)

        else:

            boolean.append(False)

    is_y = pd.Series(boolean)

    df_y = df[is_y]

    

    x  = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    itrate = len(df_y.index)

    for i in range(itrate):

        plt.plot(x, df_y.iloc[i,2:14], label=df_y.iloc[i,0])

        #plt.plot(x, df.iloc[112,2:14], label=df.iloc[112,0])

        plt.plot()



    plt.xlabel("Months")

    plt.ylabel("Rain data")

    plt.title("Line Graph of rainfall in "+str(y))

    #plt.legend()

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    plt.show()