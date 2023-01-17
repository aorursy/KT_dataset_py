import pandas as pd 

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/military-expenditure-of-countries-19602019/Military Expenditure.csv',index_col = 0)

df.info()
df.head()
df_Russia = df.loc[['Russian Federation'],:]

df_Russia.head()
x = []

y = []

for entry in df_Russia:

    for i in df_Russia[entry]:

        if entry.isdigit() and int(entry)>=1993:

            x.append(int(entry))

            y.append(int(i))

plt.plot(x,y,color="red")

plt.title('Russian military spending')

plt.xlabel('Year')

plt.ylabel('Spending')

plt.xticks([1995,2000,2005,2010,2015])

plt.yticks([20000000000,40000000000,60000000000,80000000000],['20B','40B','60B','80B'])

plt.show()

            