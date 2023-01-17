import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.getcwd())

print(os.listdir())
df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")

df.head(5)
df.isnull().any().sum()
from datetime import datetime

df['Date'] = pd.to_datetime(df['Date'],format="%d/%m/%Y %H:%M:%S %p")
import seaborn as sns

from matplotlib import pyplot as plt
sdf = df[['Value','State']].groupby(['State'], as_index=False).sum()

sdf = sdf.sort_values('Value',ascending=False)

sdf.head()
g=sns.barplot(y='State',x='Value',data=sdf)

plt.title("Statewise Border Crossing")

plt.xlabel("Count of People Entering US")
sdf = df[['Value','Port Name']].groupby(['Port Name'], as_index=False).sum()

sdf = sdf.sort_values('Value',ascending=False)

sdf.head()
g=sns.barplot(y='Port Name',x='Value',data=sdf.iloc[:10,])

plt.title("Top 10 Ports of Entry")

plt.xlabel("Count of People Entering US")
df['Year'] = df['Date'].dt.year

sdf = df[['Year','Value','Border']].groupby(['Year'], as_index=False).sum()

sdf.head()
g=sns.lmplot(x='Year',y='Value',ci=False,robust=True,data=sdf)

plt.ticklabel_format(style='plain', axis='y')

plt.title("Number of People coming into US is decreasing with Time")
sdf = df[['Year','Value','Border']].groupby(['Year','Border'], as_index=False).sum()

sdf.head()
g=sns.lmplot(x='Year',y='Value',ci=False,robust=True,hue='Border',data=sdf)

plt.ticklabel_format(style='plain', axis='y')

plt.title("People coming into US are mostly from Mexico")