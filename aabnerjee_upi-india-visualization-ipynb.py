#Import the required libararies



import requests

import pandas as pd

import datetime

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
import pandas as pd

df = pd.read_csv("../input/UPI_Data_FEB_2020.csv")
#Drop the unwanted columns

df = df.drop(columns = "Unnamed: 0")

df['month'] = df['month'].astype('datetime64[ns]')
df.head(5)
df.describe()
df.info()
fig, ax = plt.subplots()

ax.plot(df['month'], df['amount'])

ax.set(xlabel='Months', ylabel='Transaction Amount (in cr)',

       title='UPI Amount wise growth')

ax.grid()

fig.savefig("test.png")

plt.show()



fig, ax = plt.subplots()

ax.plot(df['month'], df['volume'])

ax.set(xlabel='Months', ylabel='Transaction Volume (in Mn)',

       title='UPI Volume wise growth')

ax.grid()

fig.savefig("test.png")

plt.show()


