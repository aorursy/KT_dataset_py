import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime, time
df = pd.read_csv("../input/SPY-2.csv")

print(df.head())
df['Date'] = df['Date'].apply(lambda date_str: datetime.datetime.strptime(date_str, '%Y-%m-%d').timestamp())

print(df.head())
sns.regplot(data=df, x='Date', y='Close', ci=None)

plt.show()
threshold = 1420070400

five_years = df[df['Date'] > threshold].dropna()



sns.regplot(data=five_years, x='Date', y='Close', ci=None)

plt.show()
threshold = 1483228800

three_years = df[df['Date'] > threshold].dropna()



sns.regplot(data=three_years, x='Date', y='Close', ci=None)

plt.show()
threshold = 1514764800

one_year = df[df['Date'] > threshold].dropna()



sns.regplot(data=one_year, x='Date', y='Close', ci=None)

plt.show()